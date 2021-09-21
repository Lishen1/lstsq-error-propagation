from typing import Callable
from scipy.linalg import block_diag
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from numpy.core.function_base import linspace
from parse import parse

from scipy.ndimage.measurements import label
import transforms3d as t3d
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


def confidence_ellipse(cov, ax, n_std=3.0, facecolor='none', **kwargs):

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.
    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.3*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def error_ellipse(ax, xc, yc, cov, sigma=1, **kwargs):
    '''
    Plot an error ellipse contour over your data.
    Inputs:
    ax : matplotlib Axes() object
    xc : x-coordinate of ellipse center
    yc : x-coordinate of ellipse center
    cov : covariance matrix
    sigma : # sigma to plot (default 1)
    additional kwargs passed to matplotlib.patches.Ellipse()
    '''
    w, v = np.linalg.eigh(cov) # assumes symmetric matrix
    order = w.argsort()[::-1]
    w, v = w[order], v[:,order]
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    ellipse = Ellipse(xy=(xc,yc),
                    width=2.*sigma*np.sqrt(w[0]),
                    height=2.*sigma*np.sqrt(w[1]),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    ax.add_patch(ellipse)

def project_point(R, t, K, point):
    uv = K@R.T@(point.reshape(3, 1)-t.reshape(3, 1))
    uv /= uv[2]
    return uv[:2]

def make_A(r_arr):
    N = len(r_arr)
    A = np.zeros((N*3, 3))
    for i, r in enumerate(r_arr):
        A[i*3:i*3+3, :] = np.eye(3) - r@r.T

    return A

def make_b(o_arr, r_arr):
    N = len(r_arr)
    b = np.zeros((N*3, 1))
    for i, r in enumerate(r_arr):
        A = np.eye(3) - r@r.T
        b[i*3:i*3+3, :] = A@o_arr[i].reshape(3, 1)

    return b

def pix2ray(R_arr, uv_arr, K):
    r_vec = []

    iK = np.linalg.inv(K)

    N = len(uv_arr)

    for i, uv in enumerate(uv_arr):
        R = R_arr[i]
        r = iK@uv
        r /= np.linalg.norm(r)
        r = R@r
        r = r.reshape(3, 1)
        r_vec.append(r)

    return r_vec


def pix2ray(R_arr, uv_arr, K):
    r_vec = []

    iK = np.linalg.inv(K)

    N = len(uv_arr)

    for i, uv in enumerate(uv_arr):
        R = R_arr[i]
        r = iK@uv
        r /= np.linalg.norm(r)
        r = R@r
        r = r.reshape(3, 1)
        r_vec.append(r)

    return r_vec

def triangulate_mp_from_rays(o_arr, r_arr):
    A = make_A(r_arr)
    b = make_b(o_arr, r_arr)

    res = (np.linalg.inv(A.T@A)@A.T@b).reshape(3, 1)
    
    return res

def triangulate_mp_from_rays_fun(x: np.ndarray, t_arr, K):

    rays = []

    for i in range(int(len(x)/3)):
        rays.append(np.array([x[i*3+0], x[i*3+1], x[i*3+2]]).reshape(-1, 1))

    point = triangulate_mp_from_rays(t_arr, rays)

    return point

def gen_data(pix_err_sig, ref_point, N, fx, fy, w, h, max_parallax):
    cx = w/2
    cy = h/2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    
    dist = np.linalg.norm(ref_point)
    side = 2*np.tan(np.deg2rad(max_parallax)/2)*dist

   

    R_arr = []
    t_arr = [] 
    uv_arr = []
    da = 0.0
    for dx in np.linspace(0, side, N):
        R = t3d.euler.euler2mat(-da, 0, np.sin(da))
        t = np.array([dx, np.sin(dx*0.1)*10*0, 0.0]).reshape(3, 1)

        uv = K@R.T@(ref_point-t)
        uv/=uv[2]

        uv[:2] += np.random.normal(0, pix_err_sig, 2).reshape(-1, 1)

        R_arr.append(R)
        t_arr.append(t)
        uv_arr.append(uv)

        da += 0.001

    return R_arr,t_arr, uv_arr, K


def linear_error_propagation(mean: np.ndarray, cov: np.ndarray, func: Callable, fjac: Callable = None):

    J = fjac(mean)
    Py = J@cov@J.T
    ym = func(mean)

    return ym, Py

def lstsq_sol_Jac(A:np.ndarray, b:np.ndarray, JA:np.ndarray, Jb:np.ndarray):
    xsize = JA.shape[1]

    ATA = A.T@A
    ATb = A.T@b

    invATA = np.linalg.inv(ATA)

    rc = A.shape[0]
    cc = A.shape[1]

    JinvATAATb = np.zeros((cc, xsize))

    for i in range(xsize):
        JAi = JA[:, i].reshape(rc, cc)
        Jbi = Jb[:, i].reshape(-1, 1)

        # JATAi = JAi.T@A+A.T@JAi

        JATAi = JAi.T@A + (JAi.T@A).T

        JATbi = JAi.T@b+A.T@Jbi

        JinvAi = -invATA@JATAi@invATA
        JinvATAATb[:, i] = (JinvAi@ATb+invATA@JATbi).flatten()

    return JinvATAATb

def proj_J(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    recZ  = 1.0 / z
    recz2 = recZ * recZ

    fjac = np.zeros((2, 3))

    fjac[0, 0] = recZ
    fjac[0, 2] = -x * recz2

    fjac[1, 1] = recZ
    fjac[1, 2] = -y * recz2

    return fjac


def trans_proj_J(mean: np.ndarray, R: np.ndarray, t: np.ndarray, fx: float, fy: float):
    pJ = proj_J(R.T@(mean.reshape(-1, 1)-t.reshape(-1, 1)).flatten())

    F = np.array([
        [fx, 0],
        [0, fy]
    ])

    return F@pJ@R.T

def pix2ray_fun_list(x: np.ndarray, R_arr, K):

    uvs = []

    for i in range(int(len(x)/2)):
        uvs.append(np.array([x[i*2+0], x[i*2+1], 1.0]))

    r_arr = pix2ray(R_arr, uvs, K)

    # print('r_arr.shape', r_arr.shape)
    return r_arr

def Jpix2ray_fun(x: np.ndarray, R_arr, K):

    uvs = []

    iK = np.linalg.inv(K)

    for i in range(int(len(x)/2)):
        uvs.append(np.array([x[i*2+0], x[i*2+1], 1.0]))

    N = len(uvs)

    j_arr = []
    for i, uvi in enumerate(uvs):
        R = R_arr[i]

        uv = uvi.reshape(-1, 1)

        r = iK@uv

        fnorm = np.linalg.norm(r)

        Jf = (np.eye(3)*fnorm-r@r.T/fnorm)/fnorm**2
        j = R@Jf@iK

        j_arr.append(j[:, 0:2])

    J = np.array(block_diag(*j_arr))

    return J

def Jtriangulate_mp_from_rays_fun(x, R_arr, t_arr, K):
    rays = pix2ray_fun_list(x, R_arr, K)
    
    A = make_A(rays)
    b = make_b(t_arr, rays)

    drrx = Jpix2ray_fun(x, R_arr, K)

    r = np.array(rays)

    Js = []
    for i, r in enumerate(rays):
        Jr = drrx[i*3:i*3+3, i*2:i*2+2]

        J0 = -r@(Jr.T)[0,:].reshape(1, -1) + Jr[:,0].reshape(-1, 1)@r.T
        J1 = -r@(Jr.T)[1,:].reshape(1, -1) + Jr[:,1].reshape(-1, 1)@r.T

        J = np.column_stack([J0.reshape(-1, 1), J1.reshape(-1, 1)])

        Js.append(J)

    Js2 = []
    for i, r in enumerate(rays):
        t = t_arr[i].reshape(-1, 1)
        Jr = drrx[i*3:i*3+3, i*2:i*2+2]
       

        J0 = (-r@(Jr.T)[0,:].reshape(1, -1) + Jr[:,0].reshape(-1, 1)@r.T)@t
        J1 = (-r@(Jr.T)[1,:].reshape(1, -1) + Jr[:,1].reshape(-1, 1)@r.T)@t

        J = np.column_stack([J0.reshape(-1, 1), J1.reshape(-1, 1)])

        Js2.append(J)

    JA = np.array(block_diag(*Js))
    Jb = np.array(block_diag(*Js2))

    # JA = r@drrx.T

    # print('JA', JA)
    return lstsq_sol_Jac(A, b, JA, Jb)

def montecarlo_error_propagation(ref_point, max_parallax, pix_err_sig, N, fx, fy, w, h):

    cloud = []

    for i in range(5000):
        R_arr,t_arr, uv_arr, K = gen_data(pix_err_sig, ref_point, N, fx, fy, w, h, max_parallax)

        rays = pix2ray(R_arr, uv_arr, K)
        point = triangulate_mp_from_rays(t_arr, rays)
        cloud.append(point)

    cloud = np.array(cloud).reshape(-1,3)
    # print(cloud.shape)
    pcov = np.cov(cloud.T)
    pmean = np.mean(cloud, axis=0)
    # print('pmean', pmean)

    return pmean, pcov, cloud



save_path = r'D:\datasets\cov_pred'

def exp_1():
    np.random.seed(0)

    N = 5

    w=800.0
    h=600.0

    fx = 900.0
    fy = 900.0


    pix_err_sig = 2.0
    
    ref_point = np.array([0.0, 2.5, 3.0e2]).reshape(3, 1)


    max_parallax = 10




    for max_parallax in np.linspace(1, 75, 18):

        max_parallax = np.round(max_parallax, 2)

        meas_mean, meas_cov3d, cloud = montecarlo_error_propagation(ref_point, max_parallax, pix_err_sig, N, fx, fy, w, h)


        R_arr, t_arr, uv_arr, K = gen_data(0.0, ref_point, N, fx, fy, w, h, max_parallax)


        uv_m = []

        for uv in uv_arr:
            uv_m.append(uv[0])
            uv_m.append(uv[1])

        uv_m = np.array(uv_m).flatten()
        ym, pred_cov3d = linear_error_propagation(uv_m,  np.eye(len(uv_m))*pix_err_sig**2, lambda x: triangulate_mp_from_rays_fun(x, t_arr, K), lambda x: Jtriangulate_mp_from_rays_fun(x, R_arr, t_arr, K))


        R = R_arr[-1]@t3d.euler.euler2mat(np.deg2rad(0), np.deg2rad(45), np.deg2rad(0))
        t= t_arr[-1].flatten() + np.array([0, 3.5, 3.3e2])

        proj_arr = []
        for p in cloud:
            uv = project_point(R, t, K, p)
            proj_arr.append(uv)

        proj_arr = np.array(proj_arr).reshape(-1, 2)
        meas_proj_cov = np.cov(proj_arr.T)

        proj_mean = np.mean(proj_arr, axis=0)

       
        _, pred_proj_cov = linear_error_propagation(meas_mean, pred_cov3d, lambda x: project_point(R, t, K, x), lambda x: trans_proj_J(x, R, t, fx, fy))


        _, pred_s3d, _ = np.linalg.svd(pred_cov3d)
        _, meas_s3d, _ = np.linalg.svd(meas_cov3d)

        _, pred_s2d, _ = np.linalg.svd(pred_proj_cov)
        _, meas_s2d, _ = np.linalg.svd(meas_proj_cov)

        print('predicted cov 3d', pred_s3d**0.5)
        print('measured cov 3d', meas_s3d**0.5)

        print('predicted cov 2d', pred_s2d**0.5)
        print('measured cov 2d', meas_s2d**0.5)

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color='b', label='метод Монте-Карло', alpha=0.1)

        ax.set_xlabel('x [м.]')
        ax.set_ylabel('y [м.]')
        ax.set_zlabel('z [м.]')

        title = ax.set_title(f'триангуляция ориентира.\n угол параллакса: {max_parallax} [град.]\n погрешность по осям: {pred_s3d[0]:.2f}, {pred_s3d[1]:.2f}, {pred_s3d[2]:.2f} [$\sigma$ м.]')
        plt.tight_layout()
        title.set_y(1.05)
        plt.subplots_adjust(top=0.8)


        set_axes_equal(ax)

        x, y, z = get_cov_ellipsoid(pred_cov3d, meas_mean)
        ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='r', alpha=0.9, label=r'предсказанный эллипсоид ковариации $3\sigma$')
        ax.legend(loc='upper right')

        plt.savefig(fr'{save_path}\3d_{max_parallax}.png')
        ax = plt.figure().add_subplot(111)
        ax.axis('equal')
        
        ax.scatter(proj_arr[:, 0], proj_arr[:, 1], color='b', alpha=0.1, label='метод Монте-Карло')
        ax.set_xlabel('u [пикс.]')
        ax.set_ylabel('v [пикс.]')

        error_ellipse(ax, proj_mean[0], proj_mean[1], pred_proj_cov, 3.0, ec='red', label=r'предсказанный эллипсоид ковариации $3\sigma$')

        ax.legend(loc='upper right')

        title = ax.set_title(f'проекция ориентира.\n угол параллакса: {max_parallax} [град.]\n погрешность по осям: {pred_s2d[0]:.2f}, {pred_s2d[1]:.2f} [$\sigma$ пикс.]')
        plt.tight_layout()
        title.set_y(1.05)
        plt.subplots_adjust(top=0.8)

        plt.savefig(fr'{save_path}\cov_pred\2d_{max_parallax}.png')

        plt.close()

def exp_2():
    rN = 50

    w=800.0
    h=600.0

    fx = 900.0
    fy = 900.0


    pix_err_sig = 2.0
    
    ref_point = np.array([0.0, 2.5, 3.0e2]).reshape(3, 1)


    max_parallax = 10


    for n in range(2, rN, 5):
        pred_s3d_arr = []
        meas_s3d_arr = []
        angs = []
        for max_parallax in np.linspace(10, 75, 18):

            max_parallax = np.round(max_parallax, 2)

            angs.append(max_parallax)

            meas_mean, meas_cov3d, cloud = montecarlo_error_propagation(ref_point, max_parallax, pix_err_sig, n, fx, fy, w, h)


            R_arr, t_arr, uv_arr, K = gen_data(0.0, ref_point, n, fx, fy, w, h, max_parallax)


            uv_m = []

            for uv in uv_arr:
                uv_m.append(uv[0])
                uv_m.append(uv[1])

            uv_m = np.array(uv_m).flatten()
            ym, pred_cov3d = linear_error_propagation(uv_m,  np.eye(len(uv_m))*pix_err_sig**2, lambda x: triangulate_mp_from_rays_fun(x, t_arr, K), lambda x: Jtriangulate_mp_from_rays_fun(x, R_arr, t_arr, K))

            _, pred_s3d, _ = np.linalg.svd(pred_cov3d)
            _, meas_s3d, _ = np.linalg.svd(meas_cov3d)

            pred_s3d_arr.append(pred_s3d**0.5)
            meas_s3d_arr.append(meas_s3d**0.5)

        
        ax = plt.figure().add_subplot(111)

        ax.set_title(f'погрешность триангуляции ориентира по N={n} лучам')

        ax.set_xlabel('угол параллакса [град.]')
        ax.set_ylabel(r'$\sigma$ [м.]')

        pred_s3d_arr = np.array(pred_s3d_arr)
        meas_s3d_arr = np.array(meas_s3d_arr)

        for i in range(3):
            ax.plot(angs, pred_s3d_arr[:, i], color=f'C{i}', linestyle='-', label=r'предсказанная погрешность')
            ax.plot(angs, meas_s3d_arr[:, i], color=f'C{i}', linestyle=':', label=r'измеренная погрешность')

        ax.legend()


        plt.savefig(fr'{save_path}\unc_plot_{n}.png')
        plt.close()

def exp_3():
    rN = 100

    w=800.0
    h=600.0

    fx = 900.0
    fy = 900.0


    pix_err_sig = 2.0
    
    ref_point = np.array([0.0, 2.5, 3.0e2]).reshape(3, 1)


    max_parallax = 40
    errs = []
    cnt = []
    for n in range(2, rN, 2):
       
        cnt.append(n)

        R_arr, t_arr, uv_arr, K = gen_data(0.0, ref_point, n, fx, fy, w, h, max_parallax)


        uv_m = []

        for uv in uv_arr:
            uv_m.append(uv[0])
            uv_m.append(uv[1])

        uv_m = np.array(uv_m).flatten()
        ym, pred_cov3d = linear_error_propagation(uv_m,  np.eye(len(uv_m))*pix_err_sig**2, lambda x: triangulate_mp_from_rays_fun(x, t_arr, K), lambda x: Jtriangulate_mp_from_rays_fun(x, R_arr, t_arr, K))

        _, pred_s3d, _ = np.linalg.svd(pred_cov3d)

        errs.append(np.mean(pred_s3d**0.5))

    ax = plt.figure().add_subplot(111)

    ax.set_title(f'средняя погрешность триангуляции ориентира')

    ax.set_xlabel('количество лучей')
    ax.set_ylabel(r'$\sigma$ [м.]')

    ax.plot(cnt, errs, label=r'средняя погрешность')
    ax.legend()

    plt.savefig(fr'{save_path}\unc_plot_N.png')
    plt.close()

if __name__ == "__main__":
    # exp_1()
    # exp_2()
    exp_3()