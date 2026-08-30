import math
from decimal import Decimal

##################################################################################
# CAM Design Parameters
##################################################################################

ACAM_cellwidth = 3                                   #cell size in [um] (HiPo cell size: 4.44 x 4.17)
ACAM_cellheight = 4.17

N_ACAM_inablock_X = 2                                #number of ACAM cells in each sub-array block (X direction)
N_ACAM_inablock_Y = 16                               #number of ACAM cells in each sub-array block (Y direction)
N_ACAM_inablock = N_ACAM_inablock_X * N_ACAM_inablock_Y

SA_width = 7                                         #SA size in [um] (HiPo SA size: 10 x 4.17)
SA_height = 4.17

N_SA_inablock = 16                                   #number of SA in each sub-array block

N_block_X = 256                                      #number of sub-array blocks per array (X direction)
N_block_Y = 8                                        #number of sub-array blocks per array (Y direction)
N_block = N_block_X * N_block_Y

Area_overhead_CAMdrivers = 1.1

gm_max_CAM = 10e-6
DynamicRange_gm_CAM = 16                             # 1T1R branches conductances and their range

V_driver_1T1R_CAM = 0.5                              # voltage forced on the array conductances [V]

T_search_CAM = 2e-9                                  # duration of voltage pulses over the conductances

##################################################################################
# CAM Energy and Area estimation
##################################################################################

ACAM_cell_area = ACAM_cellwidth * ACAM_cellheight
SA_cell_area = SA_width * SA_height

CAM_Xsize = (N_ACAM_inablock_X * N_block_X * ACAM_cellwidth + SA_width * N_block_X) * math.sqrt(Area_overhead_CAMdrivers)
CAM_Ysize = (N_ACAM_inablock_Y * N_block_Y * ACAM_cellheight) * math.sqrt(Area_overhead_CAMdrivers)

CAM_area = ((N_ACAM_inablock * ACAM_cell_area )\
            + (N_SA_inablock * SA_cell_area)) * (N_block) * Area_overhead_CAMdrivers

print ('CAM area is', int(CAM_area), '[um^2]')
print ('CAM X size is', int(CAM_Xsize), '[um]')
print ('CAM Y size is', int(CAM_Ysize), '[um]')

gm_mean = gm_max_CAM / 2                                 # mean conductance in the array [S] for a uniform distribution
# gm_mean = gm_max_CAM / math.sqrt(DynamicRange_gm_CAM)     # mean conductance in the array [S], alternative approximation

En_diss_CAM = (gm_mean * V_driver_1T1R_CAM) * V_driver_1T1R_CAM * (N_ACAM_inablock * N_block) * T_search_CAM

P_peakdiss_CAM = (gm_mean * V_driver_1T1R_CAM) * V_driver_1T1R_CAM * (N_ACAM_inablock * N_block)

Current_peak_CAM = (gm_mean * V_driver_1T1R_CAM) * (N_ACAM_inablock * N_block)

print ('The energy dissipation per search in the CAM is', "{0:.2E}".format(En_diss_CAM * 1e12), '[pJ]')

print ('The peak power dissipation while searching in the CAM is', "{0:.2E}".format(P_peakdiss_CAM), '[W]')

print ('The peak current while searching in the CAM is', "{0:.2E}".format(Current_peak_CAM), '[A]')

##################################################################################
# DPE Design Parameters
##################################################################################

DPE_1T1R_area = 6                                 # area of a 1T1R cell [um^2]
DPE_1T1Rpercell = 4
DPE_cell_area = DPE_1T1Rpercell * DPE_1T1R_area

DPE_numberofcolumns = 256                         # number of columns in the DPE
DPE_numberofrows = 256                            # number of rows in the DPE

V_driver_1T1R_DPE = 0.1                             # voltage forced on the array conductances [V]

gm_max_DPE = 10e-6
DynamicRange_gm_DPE = 16                          # 1T1R branches conductances and their range

T_search_DPE = 2e-9

N_input_slices = 1                                # number of slices of the input that are propagated to the DPE

Area_overhead_DPEdriversandmirrors = 1.1

##################################################################################
# DPE Energy and Area estimation
##################################################################################

DPE_area = DPE_cell_area * DPE_numberofcolumns * DPE_numberofrows * Area_overhead_DPEdriversandmirrors

print ('DPE area is', int(DPE_area), '[um^2]')
print ('DPE X size is', int(math.sqrt(DPE_area)), '[um]')
print ('DPE Y size is', int(math.sqrt(DPE_area)), '[um]')

gm_mean_DPE = gm_max_DPE / 2  /2                   
# mean conductance in the array [S] for a uniform distribution
# gm_mean_DPE = gm_mean_DPE / math.sqrt(DynamicRange_gm_DPE)    
# mean conductance in the array [S], alternative approximation 
# divided by two, we are only using either the positive or negative weight

V_driver_1T1R_DPE_mean = V_driver_1T1R_DPE / 2   # mean voltage driving the column for a uniform distribution

En_diss_DPE = (gm_mean_DPE * V_driver_1T1R_DPE_mean) * V_driver_1T1R_DPE_mean * \
                DPE_1T1Rpercell * DPE_numberofcolumns * DPE_numberofrows * T_search_DPE * N_input_slices

P_peakdiss_DPE = (gm_mean_DPE * V_driver_1T1R_DPE_mean) * V_driver_1T1R_DPE_mean * \
                DPE_1T1Rpercell * DPE_numberofcolumns * DPE_numberofrows
                
Current_peak_DPE = (gm_mean_DPE * V_driver_1T1R_DPE_mean) \
                * DPE_1T1Rpercell * DPE_numberofcolumns * DPE_numberofrows

print ('The energy dissipation per search in the DPE is', "{0:.2E}".format(En_diss_DPE * 1e12), '[pJ]')

print ('The peak power dissipation while searching in the DPE is', "{0:.2E}".format(P_peakdiss_DPE), '[W]')

print ('The peak current while searching in the DPE is', "{0:.2E}".format(Current_peak_DPE), '[A]')

##################################################################################
# Peripheral Circuits Parameters
##################################################################################

DAC_width_28nm = 100                                                      #DAC size in [um], 28nm
DAC_height_28nm = 30
N_DAC_perarray = 1

ADC_width_28nm = 150                                                      #ADC size in [um], 28nm
ADC_height_28nm = 160
N_ADC_perarray = 1

AreaScaling_28to22 = 0.81                                                 #from literature / PUMA spreadsheet

DAC_area = (DAC_width_28nm * DAC_height_28nm) * AreaScaling_28to22 * N_DAC_perarray
ADC_area = (ADC_width_28nm * ADC_height_28nm) * AreaScaling_28to22 * N_ADC_perarray

SRAM_area = 1                                                             # SRAM cell area [um^2]

N_config_bits = N_block*16                                                # number of configuration bits of one CAM array