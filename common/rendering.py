from threading import Event, Thread
import matplotlib.pyplot as plt

class Renderer:
    def __init__(self, env, processor, mode="diff", params={"prev_alpha": 1}):
        self.thread = RenderThread(env)
        self.thread.start()
        self.processor = processor
        self.mode = mode
        self.P = params

    def get(self, first=False, show=False):
        self.thread.begin_render()
        # Gym screen is in (height, width, channel) order. Transpose into torch order (channel, height, width).
        screen = self.processor(self.thread.get_screen().transpose((2, 0, 1)), self.thread.env)
        if self.mode == "current": output = screen
        elif self.mode == "diff":
            # Diff mode: take difference between current and (prev screen * prev_alpha)
            if first: self.prev_screen = screen
            output = screen - (self.prev_screen * self.P["prev_alpha"])
            if not first: self.prev_screen = screen
        if show: plt.imshow(self.to_numpy(output)); plt.show()
        return output

    def to_numpy(_, screen):
        image = screen.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if image.shape[2] == 1: image = image[:,:,0]
        return image

    def close(self):
        self.thread.stop()
        self.thread.join()


class RenderThread(Thread):
    """
    From:
        https://github.com/Rowing0914/TF_RL/blob/master/tf_rl/env/cartpole_pixel.py
    Which in turn is from:
        https://github.com/tqjxlm/Simple-DQN-Pytorch/blob/master/Pytorch-DQN-CartPole-Raw-Pixels.ipynb
    Data:
        - Observation: 3 x 400 x 600
    Usage:
        1. call env.step() or env.reset() to update env state
        2. call begin_render() to schedule a rendering task (non-blocking)
        3. call get_screen() to get the lastest scheduled result (block main thread if rendering not done)
    Sample Code:
    ```python
        # A simple test
        env = gym.make('CartPole-v0').unwrapped
        renderer = RenderThread(env)
        renderer.start()
        env.reset()
        renderer.begin_render()
        for i in range(100):
            screen = renderer.get_screen() # Render the screen
            env.step(env.action_space.sample()) # Select and perform an action
            renderer.begin_render()
            print(screen)
            print(screen.shape)
        renderer.stop()
        renderer.join()
        env.close()
    ```
    """
    def __init__(self, env):
        super(RenderThread, self).__init__(target=self.render)
        self._stop_event = Event()
        self._state_event = Event()
        self._render_event = Event()
        self.env = env

    def stop(self):
        """Stops the threads."""
        self._stop_event.set()
        self._state_event.set()

    def stopped(self):
        """Check if the thread has been stopped."""
        return self._stop_event.is_set()

    def begin_render(self):
        """Start rendering the screen."""
        self._state_event.set()

    def get_screen(self):
        """Get and output the screen image."""
        self._render_event.wait()
        self._render_event.clear()
        return self.screen

    def render(self):
        while not self.stopped():
            self._state_event.wait()
            self._state_event.clear()
            self.screen = self.env.render(mode='rgb_array')
            self._render_event.set()