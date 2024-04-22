import numpy as np
import sys
sys.path.append("C:/Users/MarwaABDERRAHIM/Medical-Detection3d-Toolkit")

from detection3d.vis.error_analysis import error_analysis
import pandas as pd

def CheckEqual(value, expected_value, threshold = 1e-5):
  diff = np.linalg.norm(value - expected_value)
  if diff >= threshold:
    raise AssertionError()
import csv

import csv

def load_coordinates_from_csv(csv_file):
    landmarks = {}  # Créez un dictionnaire pour stocker les données
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)  # Utilisez DictReader pour lire le fichier CSV
        for row in csvreader:
            nom = row['name']  # Obtenez la valeur de la colonne 'nom'
            x, y, z = float(row['x']), float(row['y']), float(row['z'])  # Convertissez les valeurs en flottants
            coordinates = [x, y, z]  # Créez une liste de coordonnées
            landmarks[nom] = coordinates  # Ajoutez les coordonnées au dictionnaire
    return landmarks

if __name__ == '__main__':
  # labelled_points = []
  # detected_points = []

  # labelled_points.append([0.5, 1.2, -2.3])
  # detected_points.append([0.5, 1.5, -2.7])

  # labelled_points.append([3.6, -1.8, 0.4])
  # detected_points.append([3.5, -1.8, 0.4])

  # labelled_points.append([-1.2, 2.5, 3.8])
  # detected_points.append([-1.4, 2.4, 4.0])

  # labelled_points_dict = {'fichier1': {'point_de_reperes_1': labelled_points[0], 'point_de_reperes_2': labelled_points[1], 'point_de_reperes_3': labelled_points[2]}}

  # detected_points_dict = {'fichier1': {'point_de_reperes_1': detected_points[0], 'point_de_reperes_2': detected_points[1], 'point_de_reperes_3': detected_points[2]}}

  labelled_points_dict = load_coordinates_from_csv(r"C:\Users\MarwaABDERRAHIM\Downloads\case_71_org.nii (8).csv")
  detected_points_dict = load_coordinates_from_csv(r"C:\Users\MarwaABDERRAHIM\Downloads\case_71_org.nii (8).csv")
  print("detected_points_dict", detected_points_dict)
  error_summary = error_analysis(labelled_points_dict, detected_points_dict)
  print("error_summary", error_summary)
  # CheckEqual(error_summary.min_error, 0.1)
  # CheckEqual(error_summary.max_error, 0.5)
  # CheckEqual(error_summary.mean_error, 0.3)
  # CheckEqual(error_summary.median_error, 0.3)
  # CheckEqual(error_summary.l2_norm_error_list, [0.5, 0.1, 0.3])
  # CheckEqual(error_summary.sorted_index_list, [0, 2, 1])