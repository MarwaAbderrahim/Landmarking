from collections import namedtuple
import numpy as np
import sys
sys.path.append("C:/Users/MarwaABDERRAHIM/Medical-Detection3d-Toolkit")

from detection3d.vis.gen_images import get_landmarks_stat


"""
The struct containing the error summary.
"""
ErrorSummary = namedtuple('ErrorSummary',
                          'all_cases, tp_cases tn_cases fp_cases fn_cases error_dx '
                          'error_dy error_dz error_l2 error_type error_sorted_index '
                          'mean_error_tp std_error_tp median_error_tp max_error_tp')


def error_analysis(label_landmark, detection_landmark, decending=True):
  """
  Analyze landmark detection error and return the error statistics summary.
  Input arguments:
  label_landmark: A dict whose keys and values are filenames and coordinates of labelled points respectively.
  detection_landmark: A dict whose keys and values are filenames and coordinates of detected points respectively.
  descending:          Flag indicating whether errors sorted in ascending or descending order.
  Return:
  error_summary:       Summary of error statistics.
  """
  # get the true positive files
  detected_landmarks_stat = get_landmarks_stat(detection_landmark)
  labelled_landmarks_stat = get_landmarks_stat(label_landmark)
  
  tp_cases, tn_cases, fp_cases, fn_cases = {}, {}, {}, {}
  error_dx, error_dy, error_dz, error_l2 = {}, {}, {}, {}
  mean_error_tp, std_error_tp, median_error_tp, max_error_tp = {}, {}, {}, {}
  error_sorted_index, error_type, all_cases = {}, {}, {}
  for landmark_name in detected_landmarks_stat.keys():
    tp_cases_list = list(set(detected_landmarks_stat[landmark_name]['pos']) &
                    set(labelled_landmarks_stat[landmark_name]['pos']))
    tn_cases_list = list(set(detected_landmarks_stat[landmark_name]['neg']) &
                    set(labelled_landmarks_stat[landmark_name]['neg']))
    fp_cases_list = list(set(detected_landmarks_stat[landmark_name]['pos']) &
                    set(labelled_landmarks_stat[landmark_name]['neg']))
    fn_cases_list = list(set(detected_landmarks_stat[landmark_name]['neg']) &
                    set(labelled_landmarks_stat[landmark_name]['pos']))
    
    error_dx_list, error_dy_list, error_dz_list, error_l2_list = \
      [], [], [], []
    all_file_list = []
    error_type_list = []
    for file_name in tp_cases_list:
      dx = detection_landmark[file_name][landmark_name][0] - \
           label_landmark[file_name][landmark_name][0]
      error_dx_list.append(dx)
      dy = detection_landmark[file_name][landmark_name][1] - \
           label_landmark[file_name][landmark_name][1]
      error_dy_list.append(dy)
      dz = detection_landmark[file_name][landmark_name][2] - \
           label_landmark[file_name][landmark_name][2]
      error_dz_list.append(dz)
      l2 = np.linalg.norm([dx, dy, dz])
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('TP')
      
    mean_error_tp.update({landmark_name: np.mean(error_l2_list)})
    std_error_tp.update({landmark_name: np.std(error_l2_list)})
    median_error_tp.update({landmark_name: np.median(error_l2_list)})
    max_error_tp.update({landmark_name: np.max(error_l2_list)})

    for file_name in tn_cases_list:
      dx = 0
      error_dx_list.append(dx)
      dy = 0
      error_dy_list.append(dy)
      dz = 0
      error_dz_list.append(dz)
      l2 = 0
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('TN')

    for file_name in fp_cases_list:
      dx = -1
      error_dx_list.append(dx)
      dy = -1
      error_dy_list.append(dy)
      dz = -1
      error_dz_list.append(dz)
      l2 = -1
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('FP')

    for file_name in fn_cases_list:
      dx = -1
      error_dx_list.append(dx)
      dy = -1
      error_dy_list.append(dy)
      dz = -1
      error_dz_list.append(dz)
      l2 = -1
      error_l2_list.append(l2)
      all_file_list.append(file_name)
      error_type_list.append('FN')

    all_cases.update({landmark_name: all_file_list})
    tp_cases.update({landmark_name: tp_cases_list})
    tn_cases.update({landmark_name: tn_cases_list})
    fp_cases.update({landmark_name: fp_cases_list})
    fn_cases.update({landmark_name: fn_cases_list})
    error_dx.update({landmark_name: error_dx_list})
    error_dy.update({landmark_name: error_dy_list})
    error_dz.update({landmark_name: error_dz_list})
    error_l2.update({landmark_name: error_l2_list})
    sorted_index_list = np.argsort(error_l2[landmark_name])
    if decending:
      sorted_index_list = sorted_index_list[::-1]
    error_sorted_index.update({landmark_name: sorted_index_list})
    error_type.update({landmark_name: error_type_list})

  error_summary = ErrorSummary(
    all_cases=all_cases,
    tp_cases=tp_cases,
    tn_cases=tn_cases,
    fp_cases=fp_cases,
    fn_cases=fn_cases,
    error_dx=error_dx,
    error_dy=error_dy,
    error_dz=error_dz,
    error_l2=error_l2,
    error_type=error_type,
    error_sorted_index=error_sorted_index,
    mean_error_tp=mean_error_tp,
    std_error_tp=std_error_tp,
    median_error_tp=median_error_tp,
    max_error_tp=max_error_tp
  )

  return error_summary



if __name__ == '__main__':
    # labelled_points_dict = {
          # 'fichier1':{'psisl': [150.10752868652344,323.0843811035156,274.5582580566406],
          #  'asisl':[103.4222412109375,199.06178283691406,234.26065063476562],
          #  'isl':[140.7960662841797,201.06005859375,179.5272216796875],
          #  'ps':[259.9061279706654,182.2727336778899,108.57090615976131],
          #  'isr':[376.36651611328125,214.06007385253906,190.49285888671875],
          #  'asisr':[410.6258239746094,217.06251525878906,234.81336975097656],
          #  'psisr':[337.6744079589844,341.0684509277344,269.1475830078125],
          #  'sp':[250.93182373046875,265.07275390625,207.8450164794922]} }
    labelled_points_dict = {
           'fichier1':{'psisl': [114.16641998291016,322.0600891113281,265.2004089355469],
           'asisl':[43.72390365600586,161.06007385253906,189.1954803466797],
           'isl':[82.98351287841797,188.05361938476562,138.3391571044922],
            'ps':[244.13098374606028,155.30986966171523,74.48302832719813],
            'isr':[424.473388671875,193.0609130859375,137.1870880126953],
            'asisr':[470.8290710449219,176.06617736816406,176.86355590820312],
            'psisr':[382.6529846191406,342.06005859375,250.91973876953125],
            'sp':[252.88877868652344,312.0664978027344,212.6660919189453]
            } }

    # detected_points_dict = {
    #       'fichier1':{'psisl': [251.75303273376906,360.4955201943275,123.68789447898068],
    #        'asisl':[160.8621024637252,212.20894795135544,152.25929701072798],
    #        'isl':[287.7015036625911,330.91828046886945,190.08045031294824],
    #        'ps':[173.4772113396803,253.2408721655121,52.545508565958684],
    #        'isr':[327.3954331401011,249.2252766578829,84.15401354413314],
    #        'asisr':[164.4057008328673,202.75023070008365,146.06941642196193],
    #        'psisr':[359.03356821520424,330.4955294049483,273.708035994787],
    #        'sp':[248.55929291109948,253.68357062931236,256.04461957709617]}
    # }
#     detected_points_dict = {
#             'fichier1':{'psisl': [295.1394864510991,300.0719360146251,190.89222427061839],
#             'asisl':[227.71018841718995,313.58854395355297,203.298028230525],
#              'isl':[245.15332714101905,348.5468027376073,184.24225403677244],
#              'ps':[274.92061596970234,352.6028875567059,190.6949308966063],
#              'isr':[238.0,352.39288682315333,292.447956953675],
#              'asisr':[281.51048987871377,356.0506723501049,175.44677128547804],
#              'psisr':[360.72192854439345,330.4068523717929,273.71660573581084],
#              'sp':[238.08068047741168,318.8345383886258,227.52760401451283
# ]}
#      }  
    detected_points_dict = {
             'fichier1':{'psisl': [179.23486148859956,324.51075648326133,222.9326170746434],
             'asisl':[255.9179033690392,396.9356086707985,149.5400462280022],
              'isl':[324.2286131011662,335.0330645273383,213.15815544252027],
              'ps':[132.01858438521853,194.18255547082495,100.02708447572597],
              'isr':[262.3640788628412,358.9652779572591,163.10274531390624],
              'asisr':[228.37823164806844,274.9809179768141,235.15621232057254],
              'psisr':[242.34065654450532,257.3531652777549,312.2946530198502],
              'sp':[241.87556393410782,253.89750523414756,259.39462407836817]
              }
              }



    error_summary = error_analysis(labelled_points_dict, detected_points_dict)

    # Accédez aux attributs de error_summary
    print("Mean Error TP:", error_summary.mean_error_tp)
    print("Max Error TP:", error_summary.max_error_tp)
    print("Median Error TP:", error_summary.median_error_tp)
    print("STD Error TP:", error_summary.std_error_tp)
    print("All cases:", error_summary.all_cases)
    print("TP cases", error_summary.tp_cases)
    print("TN cases", error_summary.tn_cases)
    print("FP cases", error_summary.fp_cases)
    print("FN cases", error_summary.fn_cases)
