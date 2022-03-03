import numpy as np
from scipy.stats import norm


class Interface:
    def __init__(self, pbrl): self.pbrl, self.oracle = pbrl, None
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_value, traceback): pass
    def print(self, _): pass


class VideoInterface(Interface):
    def __init__(self, pbrl): 
        Interface.__init__(self, pbrl)
        import cv2 # Lazy import
        self.mapping = {81: 1., 83: 0., 32: 0.5, 27: "esc"}

    def __enter__(self):
        self.videos = []
        for rn in self.pbrl.run_names:
            run_videos = sorted([f"video/{rn}/{f}" for f in os.listdir(f"video/{rn}") if ".mp4" in f])
            assert [int(v[-10:-4]) for v in run_videos] == list(range(len(run_videos)))
            self.videos += run_videos
        if len(self.videos) != len(self.pbrl.episodes): 
            assert len(self.videos) == len(self.pbrl.episodes) + 1
            print("Partial video found; ignoring.")                
        cv2.startWindowThread()
        cv2.namedWindow("Trajectory Pairs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Trajectory Pairs", 1000, 500)

    def __exit__(self, exc_type, exc_value, traceback): 
        cv2.destroyAllWindows()

    def __call__(self, i, j):
        vid_i = cv2.VideoCapture(self.videos[i])
        vid_j = cv2.VideoCapture(self.videos[j])
        while True:
            ret, frame1 = vid_i.read()
            if not ret: vid_i.set(cv2.CAP_PROP_POS_FRAMES, 0); _, frame1 = vid_i.read() # Will get ret = False at the end of the video, so reset.
            ret, frame2 = vid_j.read()
            if not ret: vid_j.set(cv2.CAP_PROP_POS_FRAMES, 0); _, frame2 = vid_j.read()
            if frame1 is None or frame2 is None: raise Exception("Video saving not finished!") 
            cv2.imshow("Trajectory Pairs", np.concatenate((frame1, frame2), axis=1))
            cv2.setWindowProperty("Trajectory Pairs", cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(10) & 0xFF # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1.                        
            if key in self.mapping: break
        vid_i.release(); vid_j.release()
        return self.mapping[key]


class OracleInterface(Interface):
    """
    Oracle class implementing the five modes of irrationality in the SimTeacher algorithm:

    Lee, K., L. Smith, A. Dragan, and P. Abbeel. 
    "B-Pref: Benchmarking Preference-Based Reinforcement Learning." 
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
    def __init__(self, pbrl, kind=None, oracle=None, gamma=1, sigma=0, d_skip=-np.inf, p_equal=0, epsilon=0, return_P_i=False): 
        Interface.__init__(self, pbrl)
        self.oracle = oracle
        self.gamma, self.sigma, self.d_skip, self.p_equal, self.epsilon, self.return_P_i = gamma, sigma, d_skip, p_equal, epsilon, return_P_i

    def __call__(self, i, j): 
        if type(self.oracle) == list:
            raise NotImplementedError("List-based oracle is deprecated")
            ret_i, ret_j = self.oracle[i], self.oracle[j]
        else:
            ret_i = self.myopic_sum(self.oracle(self.pbrl.episodes[i]))
            ret_j = self.myopic_sum(self.oracle(self.pbrl.episodes[j]))
        if max(ret_i, ret_j) < self.d_skip:  return "skip"
        diff = ret_i - ret_j
        if self.sigma == 0: P_i = 0.5 if diff == 0 else 1. if diff > 0 else 0.
        else:               P_i = norm.cdf(diff / self.sigma)
        if np.random.rand() <= self.epsilon: P_i = 1. - P_i
        if self.return_P_i:                  return P_i
        elif abs(P_i - 0.5) <= self.p_equal: return 0.5
        elif np.random.rand() < P_i:         return 1. 
        else:                                return 0. 

    def myopic_sum(self, rewards):
        if self.gamma == 1: return sum(rewards)
        return sum([r*(self.gamma**t) for t,r in enumerate(reversed(rewards))])