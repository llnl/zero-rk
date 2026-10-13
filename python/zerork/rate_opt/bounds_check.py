
class BoundsCheck(object):
    def __init__(self, bounds):
        self.bounds = bounds
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        ok = True
        for xi, bi in zip(x,self.bounds):
            if xi < bi[0] or xi > bi[1]:
                ok = False
                break
        return ok

