from ..common import *


@ti.data_oriented
class Engine:
    def __init__(self, res_x=512, res_y=None):
        if res_y is None:
            res_y = res_x
        self.res = tovector((res_x, res_y) if isinstance(res_x, int) else res_x)

        self.depth = ti.field(float, self.res)
        self.maxdepth = 2**25

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())

        self.bias = ti.Vector.field(2, float, ())

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            self.W2V[None] = ti.Matrix.identity(float, 4)
            self.W2V[None][2, 2] = -1
            self.V2W[None] = ti.Matrix.identity(float, 4)
            self.V2W[None][2, 2] = -1
            self.bias[None] = [0.5, 0.5]

        ti.materialize_callback(self.clear_depth)

    @ti.kernel
    def randomize_bias(self, center: ti.template()):
        if ti.static(center):
            self.bias[None] = [0.5, 0.5]
        else:
            #r = ti.sqrt(ti.random())
            #a = ti.random() * ti.tau
            #x, y = r * ti.cos(a) * 0.5 + 0.5, r * ti.sin(a) * 0.5 + 0.5
            x, y = ti.random(), ti.random()
            self.bias[None] = [x, y]

    @ti.kernel
    def render_background(self, shader: ti.template()):
        for P in ti.grouped(ti.ndrange(*self.res)):
            if self.depth[P] >= self.maxdepth:
                uv = (float(P) + self.bias[None]) / self.res * 2 - 1
                ro = mapply_pos(self.V2W[None], V(uv.x, uv.y, -1.0))
                ro1 = mapply_pos(self.V2W[None], V(uv.x, uv.y, +1.0))
                rd = (ro1 - ro).normalized()
                shader.shade_background(P, rd)

    @ti.func
    def to_viewspace(self, p):
        return mapply_pos(self.W2V[None], p)

    @ti.pyfunc
    def from_viewspace(self, p):
        return mapply_pos(self.V2W[None], p)

    @ti.func
    def to_viewport(self, p):
        return (p.xy * 0.5 + 0.5) * self.res

    @ti.func
    def from_viewport(self, p):
        return p / self.res * 2 - 1

    @ti.kernel
    def clear_depth(self):
        for P in ti.grouped(self.depth):
            self.depth[P] = (self.maxdepth << 4)

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))
