import platform
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BaseGamepad(ABC):
    @abstractmethod
    def set_left_stick(self, x: float, y: float) -> None:
        pass

    @abstractmethod
    def set_right_stick(self, x: float, y: float) -> None:
        pass

    @abstractmethod
    def set_triggers(self, lt: float, rt: float) -> None:
        pass

    @abstractmethod
    def set_button(self, name: str, pressed: bool) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class MockGamepad(BaseGamepad):
    def __init__(self):
        self._state = {
            "left_stick": (0.0, 0.0),
            "right_stick": (0.0, 0.0),
            "triggers": (0.0, 0.0),
            "buttons": {},
        }
        logger.info("MockGamepad initialized (fallback mode)")

    def set_left_stick(self, x: float, y: float) -> None:
        self._state["left_stick"] = (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))

    def set_right_stick(self, x: float, y: float) -> None:
        self._state["right_stick"] = (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))

    def set_triggers(self, lt: float, rt: float) -> None:
        self._state["triggers"] = (max(0.0, min(1.0, lt)), max(0.0, min(1.0, rt)))

    def set_button(self, name: str, pressed: bool) -> None:
        self._state["buttons"][name] = pressed

    def update(self) -> None:
        pass

    def reset(self) -> None:
        self._state = {
            "left_stick": (0.0, 0.0),
            "right_stick": (0.0, 0.0),
            "triggers": (0.0, 0.0),
            "buttons": {},
        }

    def close(self) -> None:
        logger.info("MockGamepad closed")

    def get_state(self) -> Dict[str, Any]:
        return self._state.copy()


class WindowsGamepad(BaseGamepad):
    def __init__(self):
        try:
            import vgamepad
            self._gamepad = vgamepad.VX360Gamepad()
            logger.info("WindowsGamepad initialized with ViGEmBus")
        except ImportError:
            raise RuntimeError("vgamepad not installed. Install with: pip install vgamepad")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ViGEmBus: {e}")

    def set_left_stick(self, x: float, y: float) -> None:
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        self._gamepad.left_joystick_float(x, y)

    def set_right_stick(self, x: float, y: float) -> None:
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        self._gamepad.right_joystick_float(x, y)

    def set_triggers(self, lt: float, rt: float) -> None:
        lt = max(0.0, min(1.0, lt))
        rt = max(0.0, min(1.0, rt))
        self._gamepad.left_trigger_float(lt)
        self._gamepad.right_trigger_float(rt)

    def set_button(self, name: str, pressed: bool) -> None:
        btn_map = {
            "A": 0x1000, "B": 0x2000, "X": 0x4000, "Y": 0x8000,
            "LB": 0x0100, "RB": 0x0200,
            "BACK": 0x0010, "START": 0x0008,
            "LS": 0x0040, "RS": 0x0080,
            "UP": 0x0001, "DOWN": 0x0002, "LEFT": 0x0004, "RIGHT": 0x0008,
        }
        if name.upper() in btn_map:
            if pressed:
                self._gamepad.press_button(btn_map[name.upper()])
            else:
                self._gamepad.release_button(btn_map[name.upper()])

    def update(self) -> None:
        self._gamepad.update()

    def reset(self) -> None:
        self._gamepad.reset()

    def close(self) -> None:
        self._gamepad.reset()
        logger.info("WindowsGamepad closed")


class LinuxGamepad(BaseGamepad):
    def __init__(self):
        try:
            from evdev import UInput, ecodes, AbsInfo
            self._ecodes = ecodes
            caps = {
                ecodes.EV_KEY: [
                    ecodes.BTN_A, ecodes.BTN_B, ecodes.BTN_X, ecodes.BTN_Y,
                    ecodes.BTN_TL, ecodes.BTN_TR,
                    ecodes.BTN_SELECT, ecodes.BTN_START,
                    ecodes.BTN_THUMBL, ecodes.BTN_THUMBR,
                    ecodes.BTN_DPAD_UP, ecodes.BTN_DPAD_DOWN,
                    ecodes.BTN_DPAD_LEFT, ecodes.BTN_DPAD_RIGHT,
                ],
                ecodes.EV_ABS: [
                    (ecodes.ABS_X, AbsInfo(0, -32768, 32767, 16, 0, 0)),
                    (ecodes.ABS_Y, AbsInfo(0, -32768, 32767, 16, 0, 0)),
                    (ecodes.ABS_RX, AbsInfo(0, -32768, 32767, 16, 0, 0)),
                    (ecodes.ABS_RY, AbsInfo(0, -32768, 32767, 16, 0, 0)),
                    (ecodes.ABS_Z, AbsInfo(0, 0, 32767, 16, 0, 0)),
                    (ecodes.ABS_RZ, AbsInfo(0, 0, 32767, 16, 0, 0)),
                ],
            }
            self._uinput = UInput(caps, name="IntentPad Virtual Xbox 360", version=0x1)
            logger.info("LinuxGamepad initialized with uinput")
        except ImportError:
            raise RuntimeError("evdev not installed. Install with: pip install evdev")
        except PermissionError:
            raise RuntimeError("Permission denied: add user to 'input' group or run with sudo")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize uinput: {e}")

    def set_left_stick(self, x: float, y: float) -> None:
        x = int(max(-1.0, min(1.0, x)) * 32767)
        y = int(max(-1.0, min(1.0, y)) * 32767)
        self._uinput.write(self._ecodes.EV_ABS, self._ecodes.ABS_X, x)
        self._uinput.write(self._ecodes.EV_ABS, self._ecodes.ABS_Y, y)
        self._uinput.syn()

    def set_right_stick(self, x: float, y: float) -> None:
        x = int(max(-1.0, min(1.0, x)) * 32767)
        y = int(max(-1.0, min(1.0, y)) * 32767)
        self._uinput.write(self._ecodes.EV_ABS, self._ecodes.ABS_RX, x)
        self._uinput.write(self._ecodes.EV_ABS, self._ecodes.ABS_RY, y)
        self._uinput.syn()

    def set_triggers(self, lt: float, rt: float) -> None:
        lt = int(max(0.0, min(1.0, lt)) * 32767)
        rt = int(max(0.0, min(1.0, rt)) * 32767)
        self._uinput.write(self._ecodes.EV_ABS, self._ecodes.ABS_Z, lt)
        self._uinput.write(self._ecodes.EV_ABS, self._ecodes.ABS_RZ, rt)
        self._uinput.syn()

    def set_button(self, name: str, pressed: bool) -> None:
        btn_map = {
            "A": self._ecodes.BTN_A, "B": self._ecodes.BTN_B,
            "X": self._ecodes.BTN_X, "Y": self._ecodes.BTN_Y,
            "LB": self._ecodes.BTN_TL, "RB": self._ecodes.BTN_TR,
            "BACK": self._ecodes.BTN_SELECT, "START": self._ecodes.BTN_START,
            "LS": self._ecodes.BTN_THUMBL, "RS": self._ecodes.BTN_THUMBR,
            "UP": self._ecodes.BTN_DPAD_UP, "DOWN": self._ecodes.BTN_DPAD_DOWN,
            "LEFT": self._ecodes.BTN_DPAD_LEFT, "RIGHT": self._ecodes.BTN_DPAD_RIGHT,
        }
        if name.upper() in btn_map:
            self._uinput.write(self._ecodes.EV_KEY, btn_map[name.upper()], 1 if pressed else 0)
            self._uinput.syn()

    def update(self) -> None:
        self._uinput.syn()

    def reset(self) -> None:
        self.set_left_stick(0.0, 0.0)
        self.set_right_stick(0.0, 0.0)
        self.set_triggers(0.0, 0.0)
        for btn in ["A", "B", "X", "Y", "LB", "RB", "BACK", "START", "LS", "RS",
                    "UP", "DOWN", "LEFT", "RIGHT"]:
            self.set_button(btn, False)

    def close(self) -> None:
        self._uinput.close()
        logger.info("LinuxGamepad closed")


def create_gamepad() -> BaseGamepad:
    system = platform.system().lower()
    if system == "windows":
        try:
            return WindowsGamepad()
        except Exception as e:
            logger.warning(f"WindowsGamepad failed: {e}. Falling back to MockGamepad.")
            return MockGamepad()
    elif system == "linux":
        try:
            return LinuxGamepad()
        except Exception as e:
            logger.warning(f"LinuxGamepad failed: {e}. Falling back to MockGamepad.")
            return MockGamepad()
    else:
        logger.warning(f"Unsupported platform: {system}. Using MockGamepad.")
        return MockGamepad()


class GamepadEngine:
    def __init__(self, gamepad: Optional[BaseGamepad] = None):
        self._gamepad = gamepad or create_gamepad()
        self._running = False

    @property
    def gamepad(self) -> BaseGamepad:
        return self._gamepad

    def start(self) -> None:
        self._running = True
        logger.info("GamepadEngine started")

    def stop(self) -> None:
        self._running = False
        self._gamepad.reset()
        logger.info("GamepadEngine stopped")

    def close(self) -> None:
        self.stop()
        self._gamepad.close()

    def set_left_stick(self, x: float, y: float) -> None:
        self._gamepad.set_left_stick(x, y)

    def set_right_stick(self, x: float, y: float) -> None:
        self._gamepad.set_right_stick(x, y)

    def set_triggers(self, lt: float, rt: float) -> None:
        self._gamepad.set_triggers(lt, rt)

    def set_button(self, name: str, pressed: bool) -> None:
        self._gamepad.set_button(name, pressed)

    def update(self) -> None:
        self._gamepad.update()

    def get_state(self) -> Dict[str, Any]:
        if isinstance(self._gamepad, MockGamepad):
            return self._gamepad.get_state()
        return {}