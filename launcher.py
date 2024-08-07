import os
import json
import numpy as np

# creates slurm script mom.sub
def create_slurm(p, filename):
    # p - dictionary with parameters
    if p['mem'] < 1:
        mem = str(int(p['mem']*1000))+'MB'
    else:
        mem = str(p['mem'])+'GB'
    
    lines = [
    '#!/bin/bash',
    '#SBATCH --nodes='+str(p['nodes']),
    '#SBATCH --ntasks-per-node='+str(p['ntasks']),
    '#SBATCH --cpus-per-task=1',
    '#SBATCH --mem='+mem,
    '#SBATCH --time='+str(p['time'])+':00:00',
    '#SBATCH --begin=now+'+str(p['begin']),
    '#SBATCH --job-name='+str(p['name']),
    'scontrol show jobid -dd $SLURM_JOB_ID',
    'source /home/ctrsp-2024/pp2681/mom6/MOM6-examples/build/intel/env',
    'time mpirun ./MOM6',
    'mkdir -p output',
    'mv *.nc output'
    ]
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def create_MOM_override(p, filename):
    # p - dictionary of parameters
    lines = []
    for key in p.keys():
        lines.append('#override '+key+' = '+str(p[key]))
    with open(filename,'w') as fid:
        fid.writelines([ line+'\n' for line in lines])

def run_experiment(folder, hpc, parameters):
    if os.path.exists(folder):
        print('Folder '+folder+' already exists. We skip it')
        return
        print('Folder '+folder+' already exists. Delete it? (y/n)')
        if input()!='y':
            print('Experiment is not launched. Exit.')
            return
        else:
            os.system('rm -r '+folder)
    os.system('mkdir -p '+folder)
    
    create_slurm(hpc, os.path.join(folder,'mom.sub'))
    create_MOM_override(parameters, os.path.join(folder,'MOM_override'))
    
    os.system('cp -r /home/ctrsp-2024/pp2681/notebooks/double_gyre/* '+folder)
    os.system('cp /home/ctrsp-2024/pp2681/mom6/MOM6-examples/build/intel/ocean_only/repro/MOM6 '+folder)

    with open(os.path.join(folder,'args.json'), 'w') as f:
        json.dump(parameters, f, indent=2)
    
    os.system('cd '+folder+'; sbatch mom.sub')

#########################################################################################
class dictionary(dict):  
    def __init__(self, **kw):  
        super().__init__(**kw)
    def add(self, **kw): 
        d = self.copy()
        d.update(kw)
        return dictionary(**d)
    def __add__(self, d):
        return self.add(**d)
    
def configuration(exp='R4'):
    if exp=='R2':
        return dictionary(
            NIGLOBAL=44,
            NJGLOBAL=40,
            DT=2160.,
            DT_FORCING=2160.
        )
    if exp=='R2-FGR-2':
        return dictionary(
            NIGLOBAL=36,
            NJGLOBAL=32,
            DT=2160.,
            DT_FORCING=2160.
        )
    if exp=='R2-FGR-sqrt12':
        return dictionary(
            NIGLOBAL=62,
            NJGLOBAL=56,
            DT=2160.,
            DT_FORCING=2160.
        )
    if exp=='R3':
        return dictionary(
            NIGLOBAL=66,
            NJGLOBAL=60,
            DT=1440.,
            DT_FORCING=1440.
        )
    if exp=='R4':
        return dictionary(
            NIGLOBAL=88,
            NJGLOBAL=80,
            DT=1080.,
            DT_FORCING=1080.
        )
    if exp=='R4-FGR-2':
        return dictionary(
            NIGLOBAL=72,
            NJGLOBAL=66,
            DT=1080.,
            DT_FORCING=1080.
        )
    if exp=='R4-FGR-sqrt12':
        return dictionary(
            NIGLOBAL=124,
            NJGLOBAL=114,
            DT=1080.,
            DT_FORCING=1080.
        )
    if exp=='R5':
        return dictionary(
            NIGLOBAL=110,
            NJGLOBAL=100,
            DT=1080.,
            DT_FORCING=1080.
        )
    if exp=='R6':
        return dictionary(
            NIGLOBAL=132,
            NJGLOBAL=120,
            DT=720.,
            DT_FORCING=720.
        )
    if exp=='R7':
        return dictionary(
            NIGLOBAL=154,
            NJGLOBAL=140,
            DT=720.,
            DT_FORCING=720.
        )
    if exp=='R8':
        return dictionary(
            NIGLOBAL=176,
            NJGLOBAL=160,
            DT=540.,
            DT_FORCING=540.
        )
    if exp=='R8-FGR-2':
        return dictionary(
            NIGLOBAL=144,
            NJGLOBAL=130,
            DT=540.,
            DT_FORCING=540.
        )
    if exp=='R8-FGR-sqrt12':
        return dictionary(
            NIGLOBAL=248,
            NJGLOBAL=226,
            DT=540.,
            DT_FORCING=540.
        )
    if exp=='R10':
        return dictionary(
            NIGLOBAL=220,
            NJGLOBAL=200,
            DT=432.,
            DT_FORCING=432.
        )
    if exp=='R12':
        return dictionary(
            NIGLOBAL=264,
            NJGLOBAL=240,
            DT=360.,
            DT_FORCING=360.
        )
    if exp=='R16':
        return dictionary(
            NIGLOBAL=352,
            NJGLOBAL=320,
            DT=270.,
            DT_FORCING=270.
        )
    if exp=='R32':
        return dictionary(
            NIGLOBAL=704,
            NJGLOBAL=640,
            DT=135.,
            DT_FORCING=135.
        )
    if exp=='R64':
        return dictionary(
            NIGLOBAL=1408,
            NJGLOBAL=1280,
            DT=67.5,
            DT_FORCING=67.5
        )

HPC = dictionary(
    nodes=1,
    ntasks=1,
    mem=2,
    time=24,
    name='mom6',
    begin='0hour'
)  

PARAMETERS = dictionary(
    DAYMAX=7300.0,
    RESTINT=1825.0,
    LAPLACIAN='False',
    BIHARMONIC='True',
    SMAGORINSKY_AH='True',
    SMAG_BI_CONST=0.06
) + configuration('R4')

JansenHeld = dictionary(
    USE_MEKE='True',
    MEKE_VISCOSITY_COEFF_KU=-0.15, #See Jansen2019, top of page 2852
    RES_SCALE_MEKE_VISC=False, # Do not turn off parameterization if deformation radius is resolved
    MEKE_ADVECTION_FACTOR=1.0, # YES advection of MEKE
    MEKE_BACKSCAT_RO_POW=0.0, # Turn off Klower correction for MEKE source
    MEKE_USCALE=0.1, # velocity scale in bottom drag, see Eq. 9 in Jansen2019
    # MEKE_CDRAG is responsible for dissipation of MEKE near the bottom and automatically
    # will be chosen as 0.003, which is 10 smaller than the value in Jansen2019
    MEKE_GMCOEFF=-1.0, # No GM contribution
    MEKE_KHCOEFF=0.15, # Compute diffusivity from MEKE, with same parameter as for backscatter
    MEKE_FRCOEFF=1.0, # Conersion of dissipated KE to MEKE
    MEKE_KHMEKE_FAC=1.0, # diffusivity of MEKE is defined by the diffusivity coefficient
    MEKE_KH=0.0, # backgorund diffusivity of MEKE
    MEKE_CD_SCALE=1.0, # No intensification on the surface
    MEKE_CB=0.0,
    MEKE_CT=0.0,
    MEKE_MIN_LSCALE=True,
    MEKE_ALPHA_RHINES=1.0,
    MEKE_ALPHA_GRID=1.0,
    MEKE_COLD_START=True,
    MEKE_TOPOGRAPHIC_BETA=1.0,

    LAPLACIAN=True, # Allow Laplacian operator for backscattering
    KH=0.0, # No background diffusivity
    KH_VEL_SCALE=0.0, # No velocity scale to calculate diffusivity
    SMAGORINSKY_KH=False, # No Smagorinsky diffusivity
    BOUND_KH=False, # bounding is not needed for negative diffusivity
    BETTER_BOUND_KH=False
)

#########################################################################################
if __name__ == '__main__':
    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for reduce in [0,1]:
    #         for ssm in ['False', 'True']:
    #             for reynolds in ['False', 'True']:
    #                 for zelong in ['False', 'True']:
    #                         parameters = PARAMETERS.add(
    #                             SMAG_BI_CONST=1.0,
    #                             USE_PG23='True',
    #                             PG23_REDUCE=reduce,
    #                             PG23_SSM=ssm,
    #                             PG23_REYNOLDS=reynolds,
    #                             PG23_ZELONG_DYNAMIC=zelong,
    #                             PG23_BOUNDARY_DISCARD=10,
    #                             THICKNESSDIFFUSE=True,
    #                             PG23_THICKNESS_STREAMFUN_SSM=True
    #                             ).add(**configuration(conf))
    #                         ntasks = dict(R2=4, R3=8, R4=16, R5=16, R6=16, R7=32, R8=32)[conf]
    #                         hpc = HPC.add(ntasks=ntasks, time=24)
    #                         run_experiment(f'/home/ctrsp-2024/pp2681/experiments/SSM-streamfunction/zelong-{zelong}-ssm-{ssm}-reynolds-{reynolds}-reduce-{reduce}/{conf}', hpc, parameters)

    for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        ntasks = dict(R2=4, R3=8, R4=16, R5=16, R6=16, R7=32, R8=32)[conf]
        hpc = HPC.add(ntasks=ntasks, time=24)
        # parameters = PARAMETERS.add(
        #                             SMAG_BI_CONST=1.0,
        #                             USE_PG23='True',
        #                             PG23_REDUCE=0,
        #                             PG23_SSM='True',
        #                             PG23_REYNOLDS='False',
        #                             PG23_ZELONG_DYNAMIC='True',
        #                             PG23_BOUNDARY_DISCARD=10,
        #                             PG23_BI_CONST_MIN=0.01,
        #                             PG23_TEST_ITER=2
        #                             ).add(**configuration(conf))
        # run_experiment(f'/home/ctrsp-2024/pp2681/experiments/generalization-FGR-sqrt12/DbMM/{conf}', hpc, parameters)

        # parameters = PARAMETERS.add(
        #                             SMAG_BI_CONST=1.0,
        #                             USE_PG23='True',
        #                             PG23_REDUCE=0,
        #                             PG23_SSM='True',
        #                             PG23_REYNOLDS='False',
        #                             PG23_ZELONG_DYNAMIC='True',
        #                             PG23_BOUNDARY_DISCARD=10,
        #                             PG23_BI_CONST_MIN=0.01,
        #                             PG23_TEST_ITER=2,
        #                             THICKNESSDIFFUSE=True,
        #                             PG23_THICKNESS_STREAMFUN_SSM=True
        #                             ).add(**configuration(conf))
        # run_experiment(f'/home/ctrsp-2024/pp2681/experiments/generalization-FGR-sqrt12/DbMMh/{conf}', hpc, parameters)

        # parameters = PARAMETERS.add(
        #                             SMAG_BI_CONST=1.0,
        #                             USE_PG23='True',
        #                             PG23_REDUCE=0,
        #                             PG23_SSM='True',
        #                             PG23_REYNOLDS='True',
        #                             PG23_ZELONG_DYNAMIC='True',
        #                             PG23_BOUNDARY_DISCARD=10,
        #                             PG23_BI_CONST_MIN=0.01,
        #                             PG23_TEST_ITER=2,
        #                             THICKNESSDIFFUSE=True,
        #                             PG23_THICKNESS_STREAMFUN_SSM=True
        #                             ).add(**configuration(conf))
        # run_experiment(f'/home/ctrsp-2024/pp2681/experiments/generalization-FGR-sqrt12/DbMMh-R/{conf}', hpc, parameters)
        parameters = PARAMETERS.add(
                                    SMAG_BI_CONST=1.0,
                                    USE_PG23='True',
                                    PG23_REDUCE=0,
                                    PG23_SSM='True',
                                    PG23_REYNOLDS='True',
                                    PG23_ZELONG_DYNAMIC='True',
                                    PG23_BOUNDARY_DISCARD=10,
                                    PG23_BI_CONST_MIN=0.01,
                                    PG23_TEST_ITER=2
                                    ).add(**configuration(conf))
        run_experiment(f'/home/ctrsp-2024/pp2681/experiments/generalization-FGR-sqrt12/DbMM-R/{conf}', hpc, parameters)

    #for conf in ['R4','R8']:
    # for conf in ['R2']:
    #     for reduce in [0,1]:
    #         for ssm in ['False', 'True']:
    #             for reynolds in ['False', 'True']:
    #                 #for zelong in ['False', 'True']:
    #                 for zelong in ['True']:
    #                     for FGR in ['2', 'sqrt12']:
    #                         if FGR == '2':
    #                             kw = {'PG23_TEST_WIDTH': 2.}
    #                         if FGR == 'sqrt12':
    #                             kw = {'PG23_TEST_ITER': 2}
    #                         parameters = PARAMETERS.add(
    #                             SMAG_BI_CONST=1.0,
    #                             USE_PG23='True',
    #                             PG23_REDUCE=reduce,
    #                             PG23_SSM=ssm,
    #                             PG23_REYNOLDS=reynolds,
    #                             PG23_ZELONG_DYNAMIC=zelong,
    #                             PG23_BOUNDARY_DISCARD=10,
    #                             **kw
    #                             ).add(**configuration(f'{conf}-FGR-{FGR}'))
    #                         ntasks = dict(R2=8,R4=32,R8=32)[conf]
    #                         hpc = HPC.add(ntasks=ntasks, time=24)
    #                         run_experiment(f'/home/ctrsp-2024/pp2681/experiments/generalization-boundary10/zelong-{zelong}-ssm-{ssm}-reynolds-{reynolds}-reduce-{reduce}/{conf}-FGR-{FGR}', hpc, parameters)

    # for conf in ['R4']:  
    #     for Cs in ['0.03', '0.06']:
    #         for CR in ['20','40','60','80']:
    #             for test_iter in [1,2]:             
    #                 parameters = PARAMETERS.add(
    #                     SMAG_BI_CONST=Cs,
    #                     USE_PG23='True',
    #                     PG23_REDUCE=1,
    #                     PG23_SSM='False',
    #                     PG23_REYNOLDS='True',
    #                     PG23_ZELONG_DYNAMIC='True',
    #                     PG23_DYNAMIC_CS='False',
    #                     PG23_CR_SET=CR,
    #                     PG23_TEST_ITER = test_iter
    #                     ).add(**configuration(f'{conf}'))
    #                 ntasks = dict(R4=16)[conf]
    #                 hpc = HPC.add(ntasks=ntasks, time=24)
    #                 run_experiment(f'/home/ctrsp-2024/pp2681/experiments/CR-CS-constant/{conf}-Cs-{Cs}-CR-{CR}-iter-{test_iter}', hpc, parameters)

    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for Cs in [0.01, 0.03, 0.06]:
    #         parameters = PARAMETERS.add(
    #             SMAG_BI_CONST=Cs,
    #             THICKNESSDIFFUSE=True,
    #             PG23_THICKNESS_STREAMFUN_SSM=True
    #             ).add(**configuration(conf))
    #         ntasks = dict(R2=4, R3=8, R4=16, R5=16, R6=16, R7=32, R8=32)[conf]
    #         hpc = HPC.add(ntasks=ntasks, time=24)
    #         run_experiment(f'/home/ctrsp-2024/pp2681/experiments/SSM-streamfunction/bare-{Cs}/{conf}', hpc, parameters)

    # for conf in ['R4']:
    #     for Cs in [0.03]:
    #         for FGR in ['2', 'sqrt12']:
    #             if FGR == '2':
    #                 kw = {'PG23_TEST_WIDTH': 2.}
    #             if FGR == 'sqrt12':
    #                 kw = {'PG23_TEST_ITER': 2}
    #             parameters = PARAMETERS.add(
    #                 SMAG_BI_CONST=Cs,
    #                 THICKNESSDIFFUSE=True,
    #                 PG23_THICKNESS_STREAMFUN_SSM=True,
    #                 **kw
    #                 ).add(**configuration(f'{conf}-FGR-{FGR}'))
    #             ntasks = dict(R2=16, R3=8, R4=16, R5=16, R6=16, R7=32, R8=32)[conf]
    #             hpc = HPC.add(ntasks=ntasks, time=24)
    #             run_experiment(f'/home/ctrsp-2024/pp2681/experiments/SSM-streamfunction/bare-{Cs}/{conf}-FGR-{FGR}', hpc, parameters)

    # for conf in ['R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
    #     for reduce in [0,1]:
    #         parameters = PARAMETERS.add(
    #             SMAG_BI_CONST=1.0,
    #             USE_PG23='True',
    #             PG23_REDUCE=reduce,
    #             PG23_SSM='True',
    #             PG23_ZELONG_DYNAMIC='True',
    #             PG23_REYNOLDS='True'
    #             ).add(**configuration(conf))
    #         ntasks = dict(R2=4, R3=8, R4=16, R5=16, R6=16, R7=32, R8=32)[conf]
    #         hpc = HPC.add(ntasks=ntasks, time=24)
    #         run_experiment(f'/home/ctrsp-2024/pp2681/experiments/generalization/zelong-three-component-reduce-{reduce}/{conf}', hpc, parameters)