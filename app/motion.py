import cv2

class MotionGate:
    def __init__(self, min_pixels: int = 5000, ratio: float = 0.0075):
        self.mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.min_pixels = min_pixels
        self.ratio = ratio

    def is_motion(self, frame) -> bool:
        fg = self.mog.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        nz = cv2.countNonZero(fg)
        H, W = frame.shape[:2]
        return nz > max(self.min_pixels, int(H * W * self.ratio))
