import xgcm
import numpy as np
from helpers.computational_tools import select_LatLon

def filter_iteration(u, mask=None, x='xq', y='yh', niter=1, filter_width=np.sqrt(6), neumann=False):
    weight_side = filter_width**2 / 24.
    weight_center = 1. - 2. * weight_side 
    def weighted_sum(x, axis):
        weight = np.array([[weight_side, weight_center, weight_side]]).T @ np.array([[weight_side, weight_center, weight_side]])
        return (x * weight).sum((-1,-2))

    uf = u
    if mask is not None:
        uf = uf * mask
    for i in range(niter):
        uf = uf.pad({x:1,y:1}, constant_values=0).rolling({x:3, y:3}, center=True).reduce(weighted_sum).fillna(0.).isel({x:slice(1,-1),y:slice(1,-1)})
        if neumann:
            maskf = mask.pad({x:1, y:1}, constant_values=0).rolling({x:3, y:3}, center=True).reduce(weighted_sum).fillna(0.).isel({x:slice(1,-1),y:slice(1,-1)})
            uf = uf / (maskf + 1e-20)
        if mask is not None:
            uf = uf * mask
    return uf.chunk({y:-1,x:-1})

def compute_velocity_gradients(u, v, static, grid):
    dudx = grid.diff(u * static.wet_u / static.dyCu, 'X') * static.dyT / static.dxT
    dvdy = grid.diff(v * static.wet_v / static.dxCv, 'Y') * static.dxT / static.dyT
    
    dudy = (grid.diff(u * static.wet_u / static.dxCu, 'Y') * static.dxBu / static.dyBu * static.wet_c)
    dvdx = (grid.diff(v * static.wet_v / static.dyCv, 'X') * static.dyBu / static.dxBu * static.wet_c)

    sh_xx = (dudx-dvdy) * static.wet
    sh_xy = (dvdx+dudy) * static.wet_c
    shear_mag = np.sqrt(sh_xy**2 + grid.interp(sh_xx**2,['X','Y'])) * static.wet_c

    vort_xy = (grid.diff(v * static.wet_v * static.dyCv, 'X') - grid.diff(u * static.wet_u * static.dxCu, 'Y')) * static.wet_c / (static.dxBu * static.dyBu)

    return sh_xx, sh_xy, shear_mag, vort_xy

def compute_vorticity_gradients(vort_xy, static, grid):
    vort_x = static.wet_v/static.dxCv * grid.diff(vort_xy,'X')
    vort_y = static.wet_u/static.dyCu * grid.diff(vort_xy,'Y')
    lap_vort = (grid.diff(vort_x * static.dyCv,'X') + grid.diff(vort_y * static.dxCu, 'Y')) * static.wet_c / (static.dxBu * static.dyBu)
    lap_vort_x = static.wet_v/static.dxCv * grid.diff(lap_vort,'X')
    lap_vort_y = static.wet_u/static.dyCu * grid.diff(lap_vort,'Y')
    return vort_x, vort_y, lap_vort, lap_vort_x, lap_vort_y

def dyn_model(_u, _v, static, h = None, tf_width=np.sqrt(6), tf_iter=1, filters_ratio=np.sqrt(2), ssm=False, reynolds=False, clip=False, Lat=(35,45), Lon=(5,15), SGS_CAu=None, SGS_CAv=None, neumann=False):
    if tf_iter>1:
        print('Not Implemented error in dyn_model')
        return
    
    grid = xgcm.Grid(static, coords={
            'X': {'center': 'xh', 'outer': 'xq'},
            'Y': {'center': 'yh', 'outer': 'yq'}},
            boundary={'X': 'fill', 'Y': 'fill'},
            fill_value = {'X': 0, 'Y': 0})

    static['wet_u']=np.floor(grid.interp(static.wet,'X'))
    static['wet_v']=np.floor(grid.interp(static.wet,'Y'))
    static['wet_c']=np.floor(grid.interp(static.wet,['X','Y']))

    dx2q = static.dxBu**2 ; dy2q = static.dyBu**2
    grid_sp_q4 = ((2.0*dx2q*dy2q) / (dx2q+dy2q))**2

    u = (_u * static.wet_u).fillna(0.).astype('float64')#.chunk({'Time': 50})
    v = (_v * static.wet_v).fillna(0.).astype('float64')#.chunk({'Time': 50})

    # Model in vorticity fluxes
    uf = filter_iteration(u, static.wet_u, 'xq', 'yh', tf_iter, tf_width, neumann=neumann)
    vf = filter_iteration(v, static.wet_v, 'xh', 'yq', tf_iter, tf_width, neumann=neumann)

    sh_xx, sh_xy, shear_mag, vort_xy = compute_velocity_gradients(u, v, static, grid)
    sh_xxf, sh_xyf, shear_magf, vort_xyf = compute_velocity_gradients(uf, vf, static, grid)

    vort_x, vort_y, lap_vort, lap_vort_x, lap_vort_y = compute_vorticity_gradients(vort_xy, static, grid)
    vort_xf, vort_yf, lap_vortf, lap_vort_xf, lap_vort_yf = compute_vorticity_gradients(vort_xyf, static, grid)

    smag_x_base = lap_vort_x * grid.interp(shear_mag * grid_sp_q4,'X')
    smag_y_base = lap_vort_y * grid.interp(shear_mag * grid_sp_q4,'Y')

    smag_x = filter_iteration(smag_x_base, static.wet_v, 'xh', 'yq', tf_iter, tf_width, neumann=neumann)
    smag_y = filter_iteration(smag_y_base, static.wet_u, 'xq', 'yh', tf_iter, tf_width, neumann=neumann)

    smag_xf = (filters_ratio)**4 * lap_vort_xf * grid.interp(shear_magf * grid_sp_q4,'X')
    smag_yf = (filters_ratio)**4 * lap_vort_yf * grid.interp(shear_magf * grid_sp_q4,'Y')

    m_x = smag_xf - smag_x
    m_y = smag_yf - smag_y

    leo_x = filter_iteration(grid.interp(u,['X','Y']) * grid.interp(vort_xy,'X'), static.wet_v, 'xh', 'yq', tf_iter, tf_width, neumann=neumann) - \
            grid.interp(uf,['X','Y']) * grid.interp(vort_xyf,'X')
    leo_x = leo_x * static.wet_v
    
    leo_y = filter_iteration(grid.interp(v,['X','Y']) * grid.interp(vort_xy,'Y'), static.wet_u, 'xq','yh', tf_iter, tf_width, neumann=neumann) - \
            grid.interp(vf,['X','Y']) * grid.interp(vort_xyf,'Y')
    leo_y = leo_y * static.wet_u

    if ssm:
        h_x_comb = filter_iteration(grid.interp(uf,['X','Y']) * grid.interp(vort_xyf,'X'), static.wet_v, 'xh', 'yq', 2, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(uf, static.wet_u, 'xq', 'yh', 2, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyf, static.wet_c, 'xq', 'yq', 2, tf_width, neumann=neumann),
                            'X')
    
        h_y_comb = filter_iteration(grid.interp(vf,['X','Y']) * grid.interp(vort_xyf,'Y'), static.wet_u, 'xq', 'yh', 2, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(vf, static.wet_v, 'xh', 'yq', 2, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyf, static.wet_c, 'xq', 'yq', 2, tf_width, neumann=neumann),
                            'Y')
        h_x_basef = filter_iteration(leo_x, static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann)
        h_y_basef = filter_iteration(leo_y, static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann)
        h_x = h_x_comb - h_x_basef
        h_y = h_y_comb - h_y_basef
    else:
        h_x = 0
        h_y = 0
        h_x_comb = 0
        h_y_comb = 0
        h_x_basef = 0
        h_y_basef = 0

    if reynolds:
        ur = u - uf
        vr = v - vf
        vort_xyr = vort_xy - vort_xyf
        
        bx_base = filter_iteration(grid.interp(ur,['X','Y']) * grid.interp(vort_xyr,'X'), static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(ur, static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                            'X')

        by_base = filter_iteration(grid.interp(vr,['X','Y']) * grid.interp(vort_xyr,'Y'), static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(vr, static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                            'Y')

        ur = uf - filter_iteration(uf, static.wet_u, 'xq', 'yh', 2, tf_width, neumann=neumann)
        vr = vf - filter_iteration(vf, static.wet_v, 'xh', 'yq', 2, tf_width, neumann=neumann)
        vort_xyr = vort_xyf - filter_iteration(vort_xyf, static.wet_c, 'xq', 'yq', 2, tf_width, neumann=neumann)

        bx_comb = filter_iteration(grid.interp(ur,['X','Y']) * grid.interp(vort_xyr,'X'), static.wet_v, 'xh', 'yq', 2, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(ur, static.wet_u, 'xq', 'yh', 2, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 2, tf_width, neumann=neumann),
                            'X')

        by_comb = filter_iteration(grid.interp(vr,['X','Y']) * grid.interp(vort_xyr,'Y'), static.wet_u, 'xq', 'yh', 2, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(vr, static.wet_v, 'xh', 'yq', 2, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 2, tf_width, neumann=neumann),
                            'Y')

        bx = bx_comb - filter_iteration(bx_base, static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann)
        by = by_comb - filter_iteration(by_base, static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann)
    else:
        bx = 0 * leo_x
        by = 0 * leo_y
        bx_base = 0 * leo_x
        by_base = 0 * leo_y

    mm = (grid.interp(m_x**2, 'Y') + grid.interp(m_y**2,'X')) * static.dxT * static.dyT
    lm = (grid.interp((leo_x-h_x)*m_x,'Y') + grid.interp((leo_y-h_y)*m_y,'X')) * static.dxT * static.dyT

    def brackets(x):
        '''
        Statistical averaging < >, in this case plane-averaging over the region
        '''
        if h is not None:
            return select_LatLon(x*h,Lat,Lon).sum(['xh', 'yh'])
        else:
            return select_LatLon(x,Lat,Lon).sum(['xh', 'yh'])

    if clip:
        Cs = (brackets(np.maximum(lm,0.0)) / brackets(mm))
    else:
        Cs = (brackets(lm) / brackets(mm)); 
    try:
        Cs['Time'] = u.Time
    except:
        Cs['time'] = u.time

    lb = (grid.interp((leo_x-h_x)*bx,'Y') + grid.interp((leo_y-h_y)*by,'X')) * static.dxT * static.dyT
    mb = (grid.interp(m_x*bx,'Y') + grid.interp(m_y*by,'X')) * static.dxT * static.dyT
    bb = (grid.interp(bx*bx,'Y') + grid.interp(by*by,'X')) * static.dxT * static.dyT

    CR = (brackets(lb) - Cs * brackets(mb)) / (brackets(bb) + 1e-40)

    # Combining the full model
    uvort_flux = smag_x_base * np.maximum(Cs, 0.)
    vvort_flux = smag_y_base * np.maximum(Cs, 0.)

    if ssm:
        uvort_flux += leo_x
        vvort_flux += leo_y
    
    if reynolds:
        uvort_flux += bx_base * np.maximum(CR, 0.)
        vvort_flux += by_base * np.maximum(CR, 0.)

    dudt = + vvort_flux
    dvdt = - uvort_flux

    if SGS_CAu is not None and ssm and reynolds:
        uvort_true = - SGS_CAv
        vvort_true = + SGS_CAu

        f_Cs = (grid.interp((uvort_true - leo_x)*smag_x_base,'Y') + grid.interp((vvort_true - leo_y)*smag_y_base,'X')) * static.dxT * static.dyT
        f_CR = (grid.interp((uvort_true - leo_x)*bx_base,'Y') + grid.interp((vvort_true - leo_y)*by_base,'X')) * static.dxT * static.dyT

        aa = (grid.interp(smag_x_base*smag_x_base,'Y') + grid.interp(smag_y_base*smag_y_base,'X')) * static.dxT * static.dyT
        ab = (grid.interp(smag_x_base*bx_base,'Y') + grid.interp(smag_y_base*by_base,'X')) * static.dxT * static.dyT
        bb = (grid.interp(bx_base*bx_base,'Y') + grid.interp(by_base*by_base,'X')) * static.dxT * static.dyT

        CR_opt = (brackets(f_CR) * brackets(aa) - brackets(f_Cs) * brackets(ab)) / (brackets(aa) * brackets(bb) - brackets(ab)**2)
        Cs_opt = (brackets(f_Cs) * brackets(bb) - brackets(f_CR) * brackets(ab)) / (brackets(aa) * brackets(bb) - brackets(ab)**2)

        uvort_opt = leo_x + smag_x_base * np.maximum(Cs_opt,0.) + bx_base * np.maximum(CR_opt, 0.)
        vvort_opt = leo_y + smag_y_base * np.maximum(Cs_opt,0.) + by_base * np.maximum(CR_opt, 0.)

        dudt_opt = vvort_opt
        dvdt_opt = - uvort_opt
    else:
        CR_opt = 0
        Cs_opt = 0
        dudt_opt = 0
        dvdt_opt = 0

    return {'u':u, 'v': v, 'uf': uf, 'vf': vf,
            'sh_xx': sh_xx, 'sh_xy': sh_xy, 'shear_mag': shear_mag, 'vort_xy': vort_xy,
            'sh_xxf': sh_xxf, 'sh_xyf': sh_xyf, 'shear_magf': shear_magf, 'vort_xyf': vort_xyf,
            'vort_x': vort_x, 'vort_y': vort_y, 'lap_vort': lap_vort, 'lap_vort_x': lap_vort_x, 'lap_vort_y': lap_vort_y,
            'vort_xf': vort_xf, 'vort_yf': vort_yf, 'lap_vortf': lap_vortf, 'lap_vort_xf': lap_vort_xf, 'lap_vort_yf': lap_vort_yf,
            'smag_x': smag_x, 'smag_y': smag_y,
            'smag_xf': smag_xf, 'smag_yf': smag_yf,
            'm_x' : m_x, 'm_y': m_y,
            'leo_x': leo_x, 'leo_y': leo_y,
            'h_x': h_x, 'h_y': h_y,
            'h_x_comb': h_x_comb, 'h_y_comb': h_y_comb,
            'h_x_basef': h_x_basef, 'h_y_basef': h_y_basef,
            'bx': bx, 'by': by,
            'bx_base': bx_base, 'by_base': by_base,
            'lm': lm, 'mm': mm,
            'lb': lb, 'mb': mb,
            'bb': bb,
            'Cs': Cs, 'CR': CR,
            'dudt': dudt, 'dvdt': dvdt,
            'Cs_opt': Cs_opt, 'CR_opt': CR_opt,
            'dudt_opt': dudt_opt, 'dvdt_opt': dvdt_opt
           }

def dyn_model_SSD(_u, _v, static, h=None, tf_width=np.sqrt(6), tf_iter=1, filters_ratio=np.sqrt(2), ssm=False, reynolds=False, clip=False, Lat=(35,45), Lon=(5,15), SGS_CAu=None, SGS_CAv=None, neumann=False):
    if tf_iter>1:
        print('Not Implemented error in dyn_model')
        print(tf_iter)
        return
    
    grid = xgcm.Grid(static, coords={
            'X': {'center': 'xh', 'outer': 'xq'},
            'Y': {'center': 'yh', 'outer': 'yq'}},
            boundary={'X': 'fill', 'Y': 'fill'},
            fill_value = {'X': 0, 'Y': 0})

    static['wet_u']=np.floor(grid.interp(static.wet,'X'))
    static['wet_v']=np.floor(grid.interp(static.wet,'Y'))
    static['wet_c']=np.floor(grid.interp(static.wet,['X','Y']))

    dx2q = static.dxBu**2 ; dy2q = static.dyBu**2
    grid_sp_q4 = ((2.0*dx2q*dy2q) / (dx2q+dy2q))**2

    u = (_u * static.wet_u).fillna(0.).astype('float64')#.chunk({'Time': 50})
    v = (_v * static.wet_v).fillna(0.).astype('float64')#.chunk({'Time': 50})

    # Model in vorticity fluxes
    uf = filter_iteration(u, static.wet_u, 'xq', 'yh', tf_iter, tf_width, neumann=neumann)
    vf = filter_iteration(v, static.wet_v, 'xh', 'yq', tf_iter, tf_width, neumann=neumann)

    sh_xx, sh_xy, shear_mag, vort_xy = compute_velocity_gradients(u, v, static, grid)
    sh_xxf, sh_xyf, shear_magf, vort_xyf = compute_velocity_gradients(uf, vf, static, grid)

    vort_x, vort_y, lap_vort, lap_vort_x, lap_vort_y = compute_vorticity_gradients(vort_xy, static, grid)
    vort_xf, vort_yf, lap_vortf, lap_vort_xf, lap_vort_yf = compute_vorticity_gradients(vort_xyf, static, grid)

    smag_x_base = lap_vort_x * grid.interp(shear_mag * grid_sp_q4,'X')
    smag_y_base = lap_vort_y * grid.interp(shear_mag * grid_sp_q4,'Y')

    smag_xf = lap_vort_xf * grid.interp(shear_magf * grid_sp_q4,'X')
    smag_yf = lap_vort_yf * grid.interp(shear_magf * grid_sp_q4,'Y')

    m_x = smag_xf
    m_y = smag_yf

    leo_x = filter_iteration(grid.interp(u,['X','Y']) * grid.interp(vort_xy,'X'), static.wet_v, 'xh', 'yq', tf_iter, tf_width, neumann=neumann) - \
            grid.interp(uf,['X','Y']) * grid.interp(vort_xyf,'X')
    leo_x = leo_x * static.wet_v
    
    leo_y = filter_iteration(grid.interp(v,['X','Y']) * grid.interp(vort_xy,'Y'), static.wet_u, 'xq','yh', tf_iter, tf_width, neumann=neumann) - \
            grid.interp(vf,['X','Y']) * grid.interp(vort_xyf,'Y')
    leo_y = leo_y * static.wet_u

    if ssm:
        h_x = filter_iteration(grid.interp(uf,['X','Y']) * grid.interp(vort_xyf,'X'), static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann) - \
              grid.interp(
                  filter_iteration(uf, static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann),
                  ['X','Y']) * \
              grid.interp(
                  filter_iteration(vort_xyf, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                  'X')
        h_y = filter_iteration(grid.interp(vf,['X','Y']) * grid.interp(vort_xyf,'Y'), static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann) - \
              grid.interp(
                  filter_iteration(vf, static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann),
                  ['X','Y']) * \
              grid.interp(
                  filter_iteration(vort_xyf, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                  'Y')
    else:
        h_x = 0
        h_y = 0

    if reynolds:
        ur = u - uf
        vr = v - vf
        vort_xyr = vort_xy - vort_xyf
        
        bx_base = filter_iteration(grid.interp(ur,['X','Y']) * grid.interp(vort_xyr,'X'), static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(ur, static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                            'X')

        by_base = filter_iteration(grid.interp(vr,['X','Y']) * grid.interp(vort_xyr,'Y'), static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(vr, static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                            'Y')
        
        ur = uf - filter_iteration(uf, static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann)
        vr = vf - filter_iteration(vf, static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann)
        vort_xyr = vort_xyf - filter_iteration(vort_xyf, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann)

        bx = filter_iteration(grid.interp(ur,['X','Y']) * grid.interp(vort_xyr,'X'), static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(ur, static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                            'X')

        by = filter_iteration(grid.interp(vr,['X','Y']) * grid.interp(vort_xyr,'Y'), static.wet_u, 'xq', 'yh', 1, tf_width, neumann=neumann) - \
                grid.interp(
                    filter_iteration(vr, static.wet_v, 'xh', 'yq', 1, tf_width, neumann=neumann),
                    ['X','Y']) * \
                grid.interp(
                    filter_iteration(vort_xyr, static.wet_c, 'xq', 'yq', 1, tf_width, neumann=neumann),
                            'Y')
    else:
        bx = 0 * leo_x
        by = 0 * leo_y  
        bx_base = 0 * leo_x
        by_base = 0 * leo_y

    mm = (grid.interp(m_x**2, 'Y') + grid.interp(m_y**2,'X')) * static.dxT * static.dyT
    lm = (grid.interp((leo_x-h_x)*m_x,'Y') + grid.interp((leo_y-h_y)*m_y,'X')) * static.dxT * static.dyT

    def brackets(x):
        '''
        Statistical averaging < >, in this case plane-averaging over the region
        '''
        if h is not None:
            return select_LatLon(x*h,Lat,Lon).sum(['xh', 'yh'])
        else:
            return select_LatLon(x,Lat,Lon).sum(['xh', 'yh'])

    if clip:
        Cs = (brackets(np.maximum(lm,0.0)) / brackets(mm))
    else:
        Cs = (brackets(lm) / brackets(mm)); 
    #Cs['Time'] = u.Time

    lb = (grid.interp((leo_x-h_x)*bx,'Y') + grid.interp((leo_y-h_y)*by,'X')) * static.dxT * static.dyT
    mb = (grid.interp(m_x*bx,'Y') + grid.interp(m_y*by,'X')) * static.dxT * static.dyT
    bb = (grid.interp(bx*bx,'Y') + grid.interp(by*by,'X')) * static.dxT * static.dyT

    CR = (brackets(lb) - Cs * brackets(mb)) / (brackets(bb) + 1e-40)

    # Combining the full model
    uvort_flux = smag_x_base * np.maximum(Cs, 0.)
    vvort_flux = smag_y_base * np.maximum(Cs, 0.)

    if ssm:
        uvort_flux += leo_x
        vvort_flux += leo_y
    
    if reynolds:
        uvort_flux += bx_base * np.maximum(CR, 0.)
        vvort_flux += by_base * np.maximum(CR, 0.)

    dudt = + vvort_flux
    dvdt = - uvort_flux

    if SGS_CAu is not None and ssm and reynolds:
        uvort_true = - SGS_CAv
        vvort_true = + SGS_CAu

        f_Cs = (grid.interp((uvort_true - leo_x)*smag_x_base,'Y') + grid.interp((vvort_true - leo_y)*smag_y_base,'X')) * static.dxT * static.dyT
        f_CR = (grid.interp((uvort_true - leo_x)*bx_base,'Y') + grid.interp((vvort_true - leo_y)*by_base,'X')) * static.dxT * static.dyT

        aa = (grid.interp(smag_x_base*smag_x_base,'Y') + grid.interp(smag_y_base*smag_y_base,'X')) * static.dxT * static.dyT
        ab = (grid.interp(smag_x_base*bx_base,'Y') + grid.interp(smag_y_base*by_base,'X')) * static.dxT * static.dyT
        bb = (grid.interp(bx_base*bx_base,'Y') + grid.interp(by_base*by_base,'X')) * static.dxT * static.dyT

        CR_opt = (brackets(f_CR) * brackets(aa) - brackets(f_Cs) * brackets(ab)) / (brackets(aa) * brackets(bb) - brackets(ab)**2)
        Cs_opt = (brackets(f_Cs) * brackets(bb) - brackets(f_CR) * brackets(ab)) / (brackets(aa) * brackets(bb) - brackets(ab)**2)

        uvort_opt = leo_x + smag_x_base * np.maximum(Cs_opt,0.) + bx_base * np.maximum(CR_opt, 0.)
        vvort_opt = leo_y + smag_y_base * np.maximum(Cs_opt,0.) + by_base * np.maximum(CR_opt, 0.)

        dudt_opt = vvort_opt
        dvdt_opt = - uvort_opt
    else:
        CR_opt = 0
        Cs_opt = 0
        dudt_opt = 0
        dvdt_opt = 0

    return {'u':u, 'v': v, 'uf': uf, 'vf': vf,
            'sh_xx': sh_xx, 'sh_xy': sh_xy, 'shear_mag': shear_mag, 'vort_xy': vort_xy,
            'sh_xxf': sh_xxf, 'sh_xyf': sh_xyf, 'shear_magf': shear_magf, 'vort_xyf': vort_xyf,
            'vort_x': vort_x, 'vort_y': vort_y, 'lap_vort': lap_vort, 'lap_vort_x': lap_vort_x, 'lap_vort_y': lap_vort_y,
            'vort_xf': vort_xf, 'vort_yf': vort_yf, 'lap_vortf': lap_vortf, 'lap_vort_xf': lap_vort_xf, 'lap_vort_yf': lap_vort_yf,
            'smag_xf': smag_xf, 'smag_yf': smag_yf,
            'm_x' : m_x, 'm_y': m_y,
            'leo_x': leo_x, 'leo_y': leo_y,
            'h_x': h_x, 'h_y': h_y,
            'bx': bx, 'by': by,
            'bx_base': bx_base, 'by_base': by_base,
            'lm': lm, 'mm': mm,
            'lb': lb, 'mb': mb,
            'bb': bb,
            'Cs': Cs, 'CR': CR,
            'dudt': dudt, 'dvdt': dvdt,
            'Cs_opt': Cs_opt, 'CR_opt': CR_opt,
            'dudt_opt': dudt_opt, 'dvdt_opt': dvdt_opt
           }