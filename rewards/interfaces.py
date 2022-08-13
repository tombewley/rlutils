from numpy import concatenate
from numpy.random import default_rng
from scipy.stats import norm


class Interface:
    def __init__(self, graph):
        self.graph, self.oracle = graph, None
        self.seed()

    def seed(self, seed=None):
        self.rng = default_rng(seed)

    def __enter__(self): pass
    def __exit__(self, exc_type, exc_value, traceback): pass
    def print(self, _): pass


class VideoInterface(Interface):

    import os, cv2 # Lazy import

    def __init__(self, graph, P):
        Interface.__init__(self, graph)
        self.mapping = {81: 1., 83: 0., 32: 0.5, 27: "esc"}

    def __enter__(self):
        self.videos = []
        raise NotImplementedError("Use run_name stored in graph")
        for rn in self.pbrl.run_names:
            run_videos = sorted([f"video/{rn}/{f}" for f in self.os.listdir(f"video/{rn}") if ".mp4" in f])
            assert [int(v[-10:-4]) for v in run_videos] == list(range(len(run_videos)))
            self.videos += run_videos
        if len(self.videos) != len(self.graph):
            assert len(self.videos) == len(self.graph) + 1
            print("Partial video found; ignoring.")                
        self.cv2.startWindowThread()
        self.cv2.namedWindow("Trajectory Pairs", self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow("Trajectory Pairs", 1000, 500)

    def __exit__(self, exc_type, exc_value, traceback): 
        self.cv2.destroyAllWindows()

    def __call__(self, i, j):
        vid_i = self.cv2.VideoCapture(self.videos[i])
        vid_j = self.cv2.VideoCapture(self.videos[j])
        while True:
            ret, frame1 = vid_i.read()
            if not ret: vid_i.set(self.cv2.CAP_PROP_POS_FRAMES, 0); _, frame1 = vid_i.read() # Will get ret = False at the end of the video, so reset.
            ret, frame2 = vid_j.read()
            if not ret: vid_j.set(self.cv2.CAP_PROP_POS_FRAMES, 0); _, frame2 = vid_j.read()
            if frame1 is None or frame2 is None: raise Exception("Video saving not finished!") 
            self.cv2.imshow("Trajectory Pairs", concatenate((frame1, frame2), axis=1))
            self.cv2.setWindowProperty("Trajectory Pairs", self.cv2.WND_PROP_TOPMOST, 1)
            key = self.cv2.waitKey(10) & 0xFF # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1.
            if key in self.mapping: break
        vid_i.release(); vid_j.release()
        return self.mapping[key]


class OracleInterface(Interface):
    """
    Oracle class implementing the five modes of irrationality in the SimTeacher algorithm. From:
        Lee, K., L. Smith, A. Dragan, and P. Abbeel. "B-Pref: Benchmarking Preference-Based Reinforcement Learning." 
        Neural Information Processing Systems (NeurIPS) (2021).

    (1) "Myopic" recency bias with discount factor gamma
    (2) Query skipping if max(ret_i, ret_j) is below d_skip
        - NOTE: This reduces the effective feedback budget
    (3) Gaussian noise with standard deviation sigma 
        - Analogous to beta in Bradley-Terry model
    (4) Random flipping of P_i with probability epsilon
    (5) Equal preference expression if abs(P_i - 0.5) is below p_equal

    NOTE: Order of implementation here: (1),(2),(3),(4),(5)
    is different to the original paper: (2),(5),(1),(3),(4).

    Additional features:
    (6) Return P_i directly rather than a sample from it - likely to improve performance as gives more information
    TODO:
    (7) Left-right bias *NEED TO RANDOMISE ORDER OF i,j AFTER SAMPLING FOR THIS TO WORK*
    """

    # Defaults
    P = {"gamma": 1, "sigma": 0, "d_skip": -float("inf"), "p_equal": 0, "epsilon": 0, "return_P_i": False}

    def __init__(self, graph, P):
        Interface.__init__(self, graph)
        self.oracle = P["oracle"]
        self.P.update(P)

    def __call__(self, i, j):
        ep_i, ep_j = self.graph.nodes[i], self.graph.nodes[j]
        ret_i = self.myopic_sum(self.oracle(ep_i["states"], ep_i["actions"], ep_i["next_states"]))
        ret_j = self.myopic_sum(self.oracle(ep_j["states"], ep_j["actions"], ep_j["next_states"]))
        if max(ret_i, ret_j) < self.P["d_skip"]:  return "skip"
        diff = ret_i - ret_j
        if self.P["sigma"] == 0: P_i = 0.5 if diff == 0 else 1. if diff > 0 else 0.
        else:                    P_i = norm.cdf(diff / self.P["sigma"])
        if self.rng.random() <= self.P["epsilon"]: P_i = 1. - P_i
        if self.P["return_P_i"]:                   return P_i
        elif abs(P_i - 0.5) <= self.P["p_equal"]:  return 0.5
        elif self.rng.random() < P_i:              return 1.
        else:                                      return 0.

    def myopic_sum(self, rewards):
        if self.P["gamma"] == 1: return sum(rewards)
        return sum([r*(self.P["gamma"]**t) for t,r in enumerate(reversed(rewards))])
