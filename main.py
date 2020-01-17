########################################################################################################## preprocessing
# reading csv
import glob
import os
from scipy import stats
from io import StringIO
import pandas
import numpy as np

if not (os.path.isdir("Data_pre")):
    os.makedirs("Data_pre")
else:
    print("Folder already done")
file_list = glob.glob("./BIRAFFE-procedure" + '/*.csv')

procedureData = pandas.read_csv(file_list[0], header=2, delimiter=";",
                                names=["TIMESTAMP", "WIDGET-TYPE", "ANS"], usecols=[0, 5, 6], engine='python')

for i in range(1, len(file_list)):
    procedureData = procedureData.append(pandas.read_csv(file_list[i], header=2, delimiter=";",
                                                         names=["TIMESTAMP", "WIDGET-TYPE", "ANS"], usecols=[0, 5, 6], engine='python'))
procedureData = procedureData[procedureData['WIDGET-TYPE'] != "emospace1"]
procedureData['TIMESTAMP'] = pandas.to_datetime(procedureData['TIMESTAMP'], unit='s')
procedureData['TIMESTAMP'] = pandas.Series(procedureData['TIMESTAMP']).dt.round("S")
print(procedureData['TIMESTAMP'])
print((len(procedureData)))
print(procedureData.head())

file_list2 = glob.glob("./BIRAFFE-biosigs" + '/*.csv')

biosigsData = pandas.read_csv(file_list2[0], delimiter=";", engine='python')

for i in range(1, len(file_list2)):
    biosigsData = biosigsData.append(pandas.read_csv(file_list[i], delimiter=";", engine='python'))

biosigsData = biosigsData.dropna()

biosigsData['TIMESTAMP'] = pandas.to_datetime(biosigsData['TIMESTAMP'], unit='s')
biosigsData['TIMESTAMP'] = pandas.Series(biosigsData['TIMESTAMP']).dt.round("S")

biosigsData = biosigsData.set_index('TIMESTAMP')
biosigsData = biosigsData.resample('1S').median()     #mean()

print(biosigsData.head())
print(len(biosigsData))

mergedData = procedureData.merge(right=biosigsData, how="inner", on='TIMESTAMP')
print(mergedData.head())
print(len(mergedData))
mergedData['ECG'] = stats.zscore(mergedData['ECG'], axis=0)
mergedData['EDA'] = stats.zscore(mergedData['EDA'], axis=0)
print(mergedData.head())
mergedData.to_csv("PreprocessedData.csv", index=False)

########################################################################################################## normalizacja


# summing 1000 records to one and getting time
# i = 0
# while i < tmp.__len__():
#     ADC_ecg = tmp.iloc[i:i + 999, 0].mean()
#     VCC = 3.3
#     G = 1100.0
#     n = 10
#     ECG = (((ADC_ecg / 2 ** n) - (1 / 2)) * VCC) / G
#     bitalino_data_list1.append(ECG)
#     #       print("ADC" + str(ADC_ecg) + "  " + str(ECG))
#
#     ADC_eda = tmp.iloc[i:i + 999, 1].mean()
#     EDA = ((ADC_eda / 2 ** n) * VCC) / 0.132
#     #       print("EDA" + str(ADC_eda) + "  " + str(EDA))
#
#     bitalino_data_list2.append(EDA)
#     bitalino_data_list_time.append(timemilis + i / 1000 - 7200)
#     i += 1000
#
# # finding minimal value of EKG and EDA
#
# j = 0
# min1 = 1000
# min2 = 1000
# while j < bitalino_data_list1.__len__():
#     ekg = bitalino_data_list1[j]
#     eda = bitalino_data_list2[j]
#     if min1 > ekg:
#         min1 = ekg
#         # print(min1)
#     if min2 > eda:
#         min2 = eda
#         # print(min2)
#     j += 1
#
# # saving data for plots
# ekg_before_normalization = bitalino_data_list1.copy()
# eda_before_normalization = bitalino_data_list2.copy()
#
# k = 0
# while k < bitalino_data_list1.__len__():
#     bitalino_data_list1[k] = (bitalino_data_list1[k] - min1) * 10000 * 8
#     bitalino_data_list2[k] = bitalino_data_list2[k] - min2
#     k += 1
#
# bitalino_modified = pandas.DataFrame(
#     {'meanEKG': bitalino_data_list1, 'meanEDA': bitalino_data_list2, 'time': bitalino_data_list_time})
# bitalino_transformed_files_list[file['title'][:4]] = bitalino_modified
# print(".", end="")
# #     print(bitalino_modified)
# except:
# print("Error when reading bitalino file " + str(file['title']))
#
# # parsing procedura
# print("\nParsing procedura")
#
# for file in procedura_file_data_list:
#     try:
#         procedura_transformed_files_list[file['title'][:4]] = pandas.read_csv(StringIO(file.GetContentString()),
#                                                                               delimiter="\t",
#                                                                               names=["personID", "numerBodzca", "war",
#                                                                                      "war", "dzwiek", "obraz",
#                                                                                      "wykorzystanyWidzet", "odpowiedz",
#                                                                                      "Czas wybory", "Timestamp"])
#         print(".", end="")
#
#     except:
#         print("\nError when reading procedura file " + str(file['title']))
#
# print("\nParsed " + str(len(bitalino_transformed_files_list)) + " bitalino files")
# print("Parsed " + str(len(procedura_transformed_files_list)) + " procedura files")
