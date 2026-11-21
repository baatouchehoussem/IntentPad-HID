import asyncio
import logging
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Set, Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .gamepad_engine import GamepadEngine, create_gamepad
from .pipeline import SignalFilter, CurveType
from .profiles import ProfileManager, Profile, AxisConfig, TriggerConfig, ButtonMap
from .adapters import HandGestureAdapter, NeuralHIDAdapter, InputAdapter

logger = logging.getLogger(__name__)


class AdapterType(str):
    HAND_GESTURE = "hand_gesture"
    NEURAL_HID = "neural_hid"


class TelemetryData(BaseModel):
    left_stick: tuple[float, float] = (0.0, 0.0)
    right_stick: tuple[float, float] = (0.0, 0.0)
    triggers: tuple[float, float] = (0.0, 0.0)
    buttons: Dict[str, bool] = {}
    active_profile: str = ""
    adapter: str = ""
    timestamp: float = 0.0


class ConfigUpdate(BaseModel):
    profile: Optional[str] = None
    adapter: Optional[str] = None
    smoothing: Optional[Dict[str, float]] = None
    deadzones: Optional[Dict[str, float]] = None
    sensitivity: Optional[Dict[str, float]] = None
    curves: Optional[Dict[str, str]] = None


class IntentPadServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._gamepad_engine = GamepadEngine()
        self._profile_manager = ProfileManager()
        self._signal_filter = self._profile_manager.create_filter()
        self._adapters: Dict[str, InputAdapter] = {}
        self._active_adapter: Optional[InputAdapter] = None
        self._active_adapter_name: str = AdapterType.HAND_GESTURE
        self._clients: Set[WebSocket] = set()
        self._running = False
        self._telemetry_task: Optional[asyncio.Task] = None
        self._control_loop_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        self._gamepad_engine.start()
        self._adapters[AdapterType.HAND_GESTURE] = HandGestureAdapter()
        self._adapters[AdapterType.NEURAL_HID] = NeuralHIDAdapter()
        await self._switch_adapter(AdapterType.HAND_GESTURE)
        self._running = True
        logger.info("IntentPadServer initialized")

    async def shutdown(self) -> None:
        self._running = False
        if self._telemetry_task:
            self._telemetry_task.cancel()
        if self._control_loop_task:
            self._control_loop_task.cancel()
        if self._active_adapter:
            self._active_adapter.stop()
        self._gamepad_engine.close()
        logger.info("IntentPadServer shutdown")

    async def _switch_adapter(self, adapter_name: str) -> bool:
        if adapter_name not in self._adapters:
            return False
        if self._active_adapter:
            self._active_adapter.stop()
        self._active_adapter = self._adapters[adapter_name]
        self._active_adapter_name = adapter_name
        self._active_adapter.start()
        self._signal_filter.reset()
        logger.info(f"Switched to adapter: {adapter_name}")
        return True

    async def _control_loop(self) -> None:
        target_dt = 1.0 / 60.0
        while self._running:
            start = time.time()
            try:
                if self._active_adapter:
                    frame = self._active_adapter.read_frame()
                    self._apply_frame(frame)
            except Exception as e:
                logger.error(f"Control loop error: {e}")

            elapsed = time.time() - start
            sleep_time = max(0, target_dt - elapsed)
            await asyncio.sleep(sleep_time)

    def _apply_frame(self, frame) -> None:
        lx, ly = self._signal_filter.process_vector(frame.left_stick[0], frame.left_stick[1], "lx", "ly")
        rx, ry = self._signal_filter.process_vector(frame.right_stick[0], frame.right_stick[1], "rx", "ry")
        lt = self._signal_filter.process_trigger("lt", frame.triggers[0])
        rt = self._signal_filter.process_trigger("rt", frame.triggers[1])

        self._gamepad_engine.set_left_stick(lx, ly)
        self._gamepad_engine.set_right_stick(rx, ry)
        self._gamepad_engine.set_triggers(lt, rt)

        active_profile = self._profile_manager.get_active_profile()
        if active_profile:
            for gesture, button in active_profile.gesture_map.items():
                if gesture in frame.buttons:
                    self._gamepad_engine.set_button(button, frame.buttons[gesture])

            for btn_name, btn_map in active_profile.button_maps.items():
                if btn_name in frame.buttons:
                    pressed = frame.buttons[btn_name]
                    if btn_map.tap:
                        self._gamepad_engine.set_button(btn_map.tap, pressed)
                    if btn_map.hold and pressed:
                        pass

        self._gamepad_engine.update()

    async def _telemetry_loop(self) -> None:
        while self._running:
            try:
                state = self._get_telemetry()
                await self._broadcast(state.model_dump_json())
            except Exception as e:
                logger.error(f"Telemetry error: {e}")
            await asyncio.sleep(1.0 / 60.0)

    def _get_telemetry(self) -> TelemetryData:
        active_profile = self._profile_manager.get_active_profile()
        return TelemetryData(
            left_stick=self._get_stick_state("left"),
            right_stick=self._get_stick_state("right"),
            triggers=self._get_trigger_state(),
            buttons=self._get_button_state(),
            active_profile=active_profile.name if active_profile else "",
            adapter=self._active_adapter_name,
            timestamp=time.time(),
        )

    def _get_stick_state(self, side: str) -> tuple[float, float]:
        if isinstance(self._gamepad_engine.gamepad, MockGamepad):
            state = self._gamepad_engine.gamepad.get_state()
            return state.get(f"{side}_stick", (0.0, 0.0))
        return (0.0, 0.0)

    def _get_trigger_state(self) -> tuple[float, float]:
        if isinstance(self._gamepad_engine.gamepad, MockGamepad):
            state = self._gamepad_engine.gamepad.get_state()
            return state.get("triggers", (0.0, 0.0))
        return (0.0, 0.0)

    def _get_button_state(self) -> Dict[str, bool]:
        if isinstance(self._gamepad_engine.gamepad, MockGamepad):
            state = self._gamepad_engine.gamepad.get_state()
            return state.get("buttons", {})
        return {}

    async def _broadcast(self, message: str) -> None:
        disconnected = set()
        for client in self._clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)
        for client in disconnected:
            self._clients.discard(client)

    async def register_client(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        logger.info(f"Client connected. Total: {len(self._clients)}")

    async def unregister_client(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self._clients)}")

    async def handle_websocket(self, websocket: WebSocket) -> None:
        await self.register_client(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    await self._handle_ws_message(msg)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            await self.unregister_client(websocket)

    async def _handle_ws_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "config_update":
            await self.update_config(ConfigUpdate(**msg.get("data", {})))
        elif msg_type == "switch_adapter":
            await self._switch_adapter(msg.get("adapter", ""))
        elif msg_type == "switch_profile":
            self._profile_manager.set_active_profile(msg.get("profile", ""))
            self._signal_filter = self._profile_manager.create_filter()
        elif msg_type == "calibrate":
            if self._active_adapter_name == AdapterType.NEURAL_HID:
                adapter = self._adapters.get(AdapterType.NEURAL_HID)
                if isinstance(adapter, NeuralHIDAdapter):
                    adapter.calibrate_center()

    async def update_config(self, config: ConfigUpdate) -> None:
        if config.profile:
            self._profile_manager.set_active_profile(config.profile)
            self._signal_filter = self._profile_manager.create_filter()

        if config.adapter:
            await self._switch_adapter(config.adapter)

        if config.smoothing:
            for axis, alpha in config.smoothing.items():
                self._signal_filter.update_config(axis, ema_alpha=alpha)

        if config.deadzones:
            for axis, dz in config.deadzones.items():
                self._signal_filter.update_config(axis, deadzone=dz)

        if config.sensitivity:
            for axis, sens in config.sensitivity.items():
                self._signal_filter.update_config(axis, sensitivity=sens)

        if config.curves:
            for axis, curve in config.curves.items():
                try:
                    self._signal_filter.update_config(axis, curve=CurveType(curve))
                except ValueError:
                    pass

    def get_video_frame(self) -> Optional[bytes]:
        if self._active_adapter_name == AdapterType.HAND_GESTURE:
            adapter = self._adapters.get(AdapterType.HAND_GESTURE)
            if isinstance(adapter, HandGestureAdapter):
                return adapter.get_mjpeg_frame()
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "active_profile": self._profile_manager._active_profile,
            "available_profiles": self._profile_manager.list_profiles(),
            "active_adapter": self._active_adapter_name,
            "available_adapters": list(self._adapters.keys()),
            "filter_config": self._signal_filter.get_config_dict(),
            "gamepad_type": type(self._gamepad_engine.gamepad).__name__,
        }


server_instance: Optional[IntentPadServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_instance
    server_instance = IntentPadServer()
    await server_instance.initialize()
    server_instance._telemetry_task = asyncio.create_task(server_instance._telemetry_loop())
    server_instance._control_loop_task = asyncio.create_task(server_instance._control_loop())
    yield
    await server_instance.shutdown()


app = FastAPI(title="IntentPad-HID", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/status")
async def status():
    if not server_instance:
        raise HTTPException(503, "Server not initialized")
    return server_instance.get_status()


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    if not server_instance:
        await websocket.close(code=1011)
        return
    await server_instance.handle_websocket(websocket)


@app.get("/video_feed")
async def video_feed():
    if not server_instance:
        raise HTTPException(503, "Server not initialized")

    async def generate():
        while True:
            frame = server_instance.get_video_frame()
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(1/30)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    if not server_instance:
        raise HTTPException(503, "Server not initialized")
    await server_instance.update_config(config)
    return {"status": "ok"}


@app.post("/api/profile/{name}")
async def switch_profile(name: str):
    if not server_instance:
        raise HTTPException(503, "Server not initialized")
    success = server_instance._profile_manager.set_active_profile(name)
    if success:
        server_instance._signal_filter = server_instance._profile_manager.create_filter()
    return {"status": "ok" if success else "error", "profile": name}


@app.post("/api/adapter/{name}")
async def switch_adapter(name: str):
    if not server_instance:
        raise HTTPException(503, "Server not initialized")
    success = await server_instance._switch_adapter(name)
    return {"status": "ok" if success else "error", "adapter": name}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()