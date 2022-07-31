import taichi as ti
from scipy.spatial.transform import Rotation as R

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-8)


@ti.func
def qrot(rot, v):
    # rot: vec4, p vec3
    qvec = ti.Vector([rot[1], rot[2], rot[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (rot[0] * uv + uuv)


@ti.func
def qrot2d(rot, v):
    return ti.Vector([rot[0] * v[0] - rot[1] * v[1], rot[1] * v[0] + rot[0] * v[1]])


@ti.pyfunc
def qmul(q, r):
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = ti.Vector([w, x, y, z])
    return out / ti.sqrt(out.dot(out))  # normalize it to prevent some unknown NaN problems.


@ti.func
def w2quat(axis_angle, dtype):
    # w = axis_angle.norm()
    w = ti.sqrt(axis_angle.dot(axis_angle) + 1e-16)
    out = ti.Vector.zero(dt=dtype, n=4)
    out[0] = 1.
    if w > 1e-9:
        v = (axis_angle / w) * ti.sin(w / 2)
        # return ti.Vector([ti.cos(w/2), v * sin(w/2)])
        out[0] = ti.cos(w / 2)
        out[1] = v[0]
        out[2] = v[1]
        out[3] = v[2]
    return out


@ti.func
def inv_trans(pos, position, rotation):
    assert rotation.norm() > 0.9
    inv_quat = ti.Vector([rotation[0], -rotation[1], -rotation[2], -rotation[3]]).normalized()
    return qrot(inv_quat, pos - position)

from pyquaternion import Quaternion
import numpy as np


def angvel(q1, q2):
    q1 = Quaternion(w=q1[0], x=q1[1], y=q1[2], z=q1[3])
    q2 = Quaternion(w=q2[0], x=q2[1], y=q2[2], z=q2[3])
    delta_q = (q2 * q1.conjugate).normalised
    delta_q_len = np.linalg.norm(delta_q.vector)
    delta_q_angle = 2 * np.arctan2(delta_q_len, delta_q.real)
    w = delta_q.vector * delta_q_angle
    return w

@ti.func
def trans(pos, position, rotation):
    return qrot(rotation.normalized(), pos) + position

def relative_rot_q(q1, q2):
    r1 = R.from_quat(np.array([q1[1], q1[2], q1[3], q1[0]]))
    r2 = R.from_quat(np.array([q2[1], q2[2], q2[3], q2[0]]))
    r_rel = (r2 * r1.inv()).as_quat()
    return np.array([r_rel[-1], r_rel[0], r_rel[1], r_rel[2]])

def relative_rot_w(q1, q2):
    r1 = R.from_quat(np.array([q1[1], q1[2], q1[3], q1[0]]))
    r2 = R.from_quat(np.array([q2[1], q2[2], q2[3], q2[0]]))
    r_rel = (r2 * r1.inv()).as_rotvec()
    return r_rel
def q_mul(q1, q2):
    r1 = R.from_quat(np.array([q1[1], q1[2], q1[3], q1[0]]))
    r2 = R.from_quat(np.array([q2[1], q2[2], q2[3], q2[0]]))
    r_rel = (r1 * r2).as_quat()
    return np.array([r_rel[-1], r_rel[0], r_rel[1], r_rel[2]])

def inverse_rot(q, v):
    r = R.from_quat(np.array([q[1], q[2], q[3], q[0]]))
    return r.inv().apply(v)


def rotate(q, v):
    r = R.from_quat(np.array([q[1], q[2], q[3], q[0]]))
    return r.apply(v)

def relative_rot(q1, q2):
    r1 = R.from_quat(np.array([q1[1], q1[2], q1[3], q1[0]]))
    r2 = R.from_quat(np.array([q2[1], q2[2], q2[3], q2[0]]))
    r_rel = (r2 * r1.inv())
    return r_rel

def quat_equal(q1, q2, thr):
    thr = thr * np.ones(4)
    return np.all(np.abs(q1-q2) < thr) or np.all(np.abs(q1+q2) < thr)