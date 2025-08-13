import math

import numpy as np
import glfw

from metadrive.utils import is_win, is_mac


def get_controller(controller_name, window=None):
    """Get the controller object.

    Args:
        controller_name: The controller name.
        window: GLFW window handle for keyboard input.

    Returns:
        The instance of controller or None if error.
    """
    controller_name = str(controller_name).lower()
    if controller_name == "keyboard":
        return KeyboardController(window)
    elif controller_name in ["xboxController", "xboxcontroller", "xbox", "gamepad", "joystick", "steering_wheel",
                             "wheel"]:
        try:
            if controller_name in ["steering_wheel", "wheel"]:
                return SteeringWheelController()
            else:
                return XboxController()
        except Exception:
            return None
    else:
        raise ValueError("No such a controller type: {}".format(controller_name))



class Controller:
    def process_input(self, vehicle):
        raise NotImplementedError

    def process_others(self, *args, **kwargs):
        pass


class KeyboardController(Controller):
    STEERING_INCREMENT = 0.04
    STEERING_DECAY = 0.25
    STEERING_INCREMENT_WHEN_INVERSE_DIRECTION = 0.25

    THROTTLE_INCREMENT = 0.1
    THROTTLE_DECAY = 0.2

    BRAKE_INCREMENT = 0.5
    BRAKE_DECAY = 0.5

    def __init__(self, window=None):
        self.window = window
        self.steering = 0.
        self.throttle_brake = 0.
        self.takeover = False
        self.np_random = np.random.RandomState(None)

    def process_input(self):
        # Use GLFW to check key states
        left_key_pressed = right_key_pressed = up_key_pressed = down_key_pressed = False

        if self.window is not None:
            left_key_pressed = glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS
            right_key_pressed = glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS
            up_key_pressed = glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS
            down_key_pressed = glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS

        # If no left or right is pressed, steering decays to the center.
        if not (left_key_pressed or right_key_pressed):
            if self.steering > 0.:
                self.steering -= self.STEERING_DECAY
                self.steering = max(0., self.steering)
            elif self.steering < 0.:
                self.steering += self.STEERING_DECAY
                self.steering = min(0., self.steering)
        elif left_key_pressed:
            if self.steering >= 0.0:  # If left is pressed and steering is in left, increment the steering a little bit.
                self.steering += self.STEERING_INCREMENT
            else:  # If left is pressed but steering is in right, steering back to left side a little faster.
                self.steering += self.STEERING_INCREMENT_WHEN_INVERSE_DIRECTION
        elif right_key_pressed:
            if self.steering <= 0.:  # If right is pressed and steering is in right, increment the steering a little
                self.steering -= self.STEERING_INCREMENT
            else:  # If right is pressed but steering is in left, steering back to right side a little faster.
                self.steering -= self.STEERING_INCREMENT_WHEN_INVERSE_DIRECTION

        # If no up or down is pressed, throttle decays to the center.
        if not (up_key_pressed or down_key_pressed):
            if self.throttle_brake > 0.:
                self.throttle_brake -= self.THROTTLE_DECAY
                self.throttle_brake = max(self.throttle_brake, 0.)
            elif self.throttle_brake < 0.:
                self.throttle_brake += self.BRAKE_DECAY
                self.throttle_brake = min(0., self.throttle_brake)
        elif up_key_pressed:
            self.throttle_brake = max(self.throttle_brake, 0.)
            self.throttle_brake += self.THROTTLE_INCREMENT
        elif down_key_pressed:
            self.throttle_brake = min(self.throttle_brake, 0.)
            self.throttle_brake -= self.BRAKE_INCREMENT

        rand = self.np_random.rand() / 10000
        self.steering += rand

        self.throttle_brake = min(max(-1., self.throttle_brake), 1.)
        self.steering = min(max(-1., self.steering), 1.)

        return [self.steering, self.throttle_brake]

    def process_others(self, takeover_callback=None):
        """This function allows the outer loop to call callback if some signal is received by the controller."""
        # Check for takeover key (T) using GLFW
        if self.window is not None and takeover_callback is not None:
            if glfw.get_key(self.window, glfw.KEY_T) == glfw.PRESS:
                takeover_callback()


class SteeringWheelController(Controller):
    RIGHT_SHIFT_PADDLE = 4
    LEFT_SHIFT_PADDLE = 5
    STEERING_MAKEUP = 1.5

    def __init__(self):
        try:
            import evdev
            from evdev import ecodes, InputDevice
        except ImportError:
            print(
                "Fail to load evdev, which is required for steering wheel control. "
                "Install evdev via pip install evdev"
            )
        try:
            import pygame
            pygame.display.init()
            pygame.joystick.init()
        except ImportError:
            print("Pygame is required for steering wheel control")
            return
        assert not is_win(), "Steering Wheel is supported in linux and mac only"
        assert pygame.joystick.get_count() > 0, "Please connect Steering Wheel or use keyboard input"
        print("Successfully Connect your Steering Wheel!")

        ffb_device = evdev.list_devices()[0]
        self.ffb_dev = InputDevice(ffb_device)

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.right_shift_paddle = False
        self.left_shift_paddle = False

        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False

        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

    def process_input(self, vehicle):
        import pygame
        from evdev import ecodes
        pygame.event.pump()
        steering = -self.joystick.get_axis(0)
        throttle_brake = -self.joystick.get_axis(2) + self.joystick.get_axis(3)
        offset = 30
        val = int(65535 * (vehicle.speed_km_h + offset) / (120 + offset)) if vehicle is not None else 0
        self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
        self.right_shift_paddle = True if self.joystick.get_button(self.RIGHT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        self.button_circle = True if self.joystick.get_button(2) else False
        self.button_rectangle = True if self.joystick.get_button(1) else False
        self.button_triangle = True if self.joystick.get_button(3) else False
        self.button_x = True if self.joystick.get_button(0) else False

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        return [steering * self.STEERING_MAKEUP, throttle_brake / 2]


class XboxController(Controller):
    """Control class for Xbox wireless controller
    Accept both wired and wireless connection
    Max steering, throttle, and break are bound by _discount.

    See https://www.pygame.org/docs/ref/joystick.html#xbox-360-controller-pygame-2-x for key mapping.
    """
    STEERING_DISCOUNT = 0.5
    THROTTLE_DISCOUNT = 0.4
    BREAK_DISCOUNT = 0.5

    BUTTON_A_MAP = 0
    BUTTON_B_MAP = 1
    BUTTON_X_MAP = 2
    BUTTON_Y_MAP = 3

    STEERING_AXIS = 0  # Left stick left-right direction.
    THROTTLE_AXIS = 3  # Right stick up-down direction.
    TAKEOVER_AXIS_2 = 4  # Right trigger
    TAKEOVER_AXIS_1 = 5  # Left trigger

    def __init__(self):
        try:
            import evdev
            from evdev import ecodes, InputDevice
        except ImportError:
            print(
                "Fail to load evdev, which is required for steering wheel control. "
                "Install evdev via pip install evdev"
            )
        try:
            import pygame
            pygame.display.init()
            pygame.joystick.init()
        except ImportError:
            print("Pygame is required for Xbox controller")
            return
        assert not is_win(), "Joystick is supported in linux and mac only"
        assert pygame.joystick.get_count() > 0, "Please connect joystick or use keyboard input"
        print("Successfully Connect your Joystick!")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.button_x = False
        self.button_y = False
        self.button_a = False
        self.button_b = False

        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

    def process_input(self, vehicle):
        import pygame
        import math
        pygame.event.pump()
        steering = -self.joystick.get_axis(self.STEERING_AXIS)
        if abs(steering) < 0.05:
            steering = 0
        elif steering < 0:
            steering = -(math.pow(2, abs(steering) * self.STEERING_DISCOUNT) - 1)
        else:
            steering = math.pow(2, abs(steering) * self.STEERING_DISCOUNT) - 1

        raw_throttle_brake = -self.joystick.get_axis(self.THROTTLE_AXIS)
        if abs(raw_throttle_brake) < 0.05:
            throttle_brake = 0
        elif raw_throttle_brake < 0:
            throttle_brake = -(math.pow(2, abs(raw_throttle_brake) * self.BREAK_DISCOUNT) - 1)
        else:
            throttle_brake = math.pow(2, abs(raw_throttle_brake) * self.THROTTLE_DISCOUNT) - 1

        self.takeover = (
            self.joystick.get_axis(self.TAKEOVER_AXIS_2) > -0.9 or self.joystick.get_axis(self.TAKEOVER_AXIS_1) > -0.9
        )

        self.button_x = True if self.joystick.get_button(self.BUTTON_X_MAP) else False
        self.button_y = True if self.joystick.get_button(self.BUTTON_Y_MAP) else False
        self.button_a = True if self.joystick.get_button(self.BUTTON_A_MAP) else False
        self.button_b = True if self.joystick.get_button(self.BUTTON_B_MAP) else False

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        return [steering, throttle_brake]
