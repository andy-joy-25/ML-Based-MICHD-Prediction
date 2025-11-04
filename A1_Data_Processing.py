import numpy as np
import os
from copy import deepcopy


class Data:
    """A data class to do all the data cleaning, allowing to create different datasets on demand."""

    def __init__(self, train_path, test_path, train_labels_path):
        # load training data and separate ID
        train = np.genfromtxt(train_path, delimiter=",", names=True)
        self.train_data = self.remove_field_name(train, "Id")
        self.train_id = train["Id"]

        # load testing data and separate ID
        test = np.genfromtxt(test_path, delimiter=",", names=True)
        self.test_data = self.remove_field_name(test, "Id")
        self.test_id = test["Id"]

        # load training labels
        self.train_labels = np.genfromtxt(
            train_labels_path, delimiter=",", skip_header=1
        )[:, 1]
        self.train_labels = np.where(
            self.train_labels == -1, 0, self.train_labels
        )  # invert labels to match [0, 1] loss functions

        # fill NAs
        self.train_data = self.fill_nas_struct(self.train_data, nan_val=-1)
        self.test_data = self.fill_nas_struct(self.test_data, nan_val=-1)

        self.ID_VARS = [
            # from first section of the pdf
            "_STATE",
            "FMONTH",
            "IDATE",
            "IMONTH",
            "IDAY",
            "IYEAR",
            "DISPCODE",
            "SEQNO",
            "_PSU",
            # from the second seciotn of the pdf - Land Line Introduction
            "CTELENUM",
            "PVTRESD1",
            "COLGHOUS",
            "STATERES",
            "CELLFON3",
            "LADULT",
            "NUMADULT",
            "NUMMEN",
            "NUMWOMEN",
            # from the third section of the pdf - Cell Phone Introduction
            "CTELNUM1",
            "CELLFON2",
            "CADULT",
            "PVTRESD2",
            "CCLGHOUS",
            "CSTATE",
            "LANDLINE",
            "HHADULT",
        ]

        # remove ID vars
        self.train_data = self.remove_field_name(self.train_data, *self.ID_VARS)
        self.test_data = self.remove_field_name(self.test_data, *self.ID_VARS)

        self.training_means = (
            dict()
        )  # means of training dataset of each column after processing to fill in the mean, initialised empty at the beginning
        return

    def remove_field_name(self, a: np.ndarray, *to_remove) -> np.ndarray:
        """removes one or more column from structured array and returns smaller array.
        Usage: remove_field_name(data, *[list with col names you want removed])

        Args:
            a: Feature matrix as structured array, shape=(N,D)
            to_remove: columns to be removed (one or more strings)

        Returns:
            a_copy: changed data.
        """
        names = list(a.dtype.names)
        names = [name for name in names if name not in to_remove]
        return a[names]

    # to overwrite category values given a pre-specified mapping, can also be used to "fill-nas" by specifiying mapping={-1: 0}
    def impute_mapping(
        self, a: np.ndarray, mapping: dict, *cols_to_apply
    ) -> np.ndarray:
        """impute a dictionary mapping to one or more columns in the structued array a. This only affect values contained as keys in the dictionary.
        Usage: impute_mapping(data, dictionary_mapping, *[list with col names you want the mapping applied])

        Args:
            a: Feature matrix as structured array, shape=(N,D)
            mapping: mapping changing the values to be applied (dict)
            cols_to_apply: columns to apply the mapping to (one or more strings)

        Returns:
            a_copy: changed data."""

        a_copy = a.copy()  # modify a copy
        for col in cols_to_apply:
            # get column
            arr = a_copy[col]

            # apply changes - this will only change the values that are present in the dictionary
            # in case this looping is too slow, we can do np.vectorize(my_dict.get)(arr), but then we have None's when the array value is not present int he dataframe,
            # so we first need to fix the dictionary to contain all values
            for k, v in mapping.items():
                arr = np.where(arr == k, v, arr)

            # overwrite old array
            a_copy[col] = arr

        return a_copy

    def convert_to_array(self, a: np.array) -> tuple[list, np.array]:
        """convert a structured np.array to a regular array

        Args:
            a: Feature matrix as structured array, shape=(N,D)

        Returns:
            names: removed feature names
            a_copy: converted array."""

        names = list(a.dtype.names)
        arrays = list()
        for col in names:
            arrays.append(a[col])

        return names, np.column_stack(arrays)

    def show_remaining_vals(self, a: np.array, val=-1):
        """Show remaining entries per column that are equal to a certain values

        Args:
            a: Feature matrix as structured array, shape=(N,D)
            val: value to be counted (float or int)
        """

        names = list(a.dtype.names)
        for col in names:
            count = (a[col] == val).sum()
            if count != 0:
                print("{} contains {} {}s".format(col, count, val))

    def show_negative_vals(self, a: np.array):
        """Print count of negative values per column

        Args:
            a: Feature matrix as structured array, shape=(N,D)
        """
        names = list(a.dtype.names)
        for col in names:
            count = (a[col] < 0).sum()
            if count != 0:
                print("{} contains {} Negatives".format(col, count))

    def fill_nas_struct(self, a: np.array, nan_val: int = -1) -> np.array:
        """Replace NAns in a structured array

        Args:
            a: Feature matrix as structured array, shape=(N,D)
            nan_val: value to replace NAns with (int)
        """
        for col in a.dtype.names:
            a[col] = np.nan_to_num(a[col], copy=True, nan=-1)
        return a

    def raw_data(
        self,
    ):
        """Method to access the raw data without any processing

        Returns:
            train_data: raw training data, shape=(N,D)
            test_data: raw testing data, shape=(N,D)
            names_train: columns names from training dataset (list)
        """
        names_train, train_data = self.convert_to_array(self.train_data)
        names_test, test_data = self.convert_to_array(self.test_data)

        # add ID back
        train_data = np.c_[self.train_id, train_data]
        test_data = np.c_[self.test_id, test_data]

        assert (
            names_train == names_test
        ), "Columns in the train and test data do not have the same ordering or contain different columns"
        return train_data, test_data, names_train

    def process_data(
        self,
    ):
        """Method to process and access the clean data

        Returns:
            train_data: processed training data, shape=(N,D)
            test_data: processed testing data, shape=(N,D)
            names_train: columns names from training dataset (list)
        """

        # clean up all individual columns
        self.train_processed = self.clean_columns(self.train_data)
        self.test_processed = self.clean_columns(self.test_data)

        # impute all the mean values in train and test with the means from the training data
        self.impute_means()

        # print remaining cols
        self.show_remaining_vals(self.train_processed, val=-1)
        self.show_negative_vals(self.train_processed)

        names_train, train_data = self.convert_to_array(self.train_processed)
        names_test, test_data = self.convert_to_array(self.test_processed)

        # add ID back
        train_data = np.c_[self.train_id, train_data]
        test_data = np.c_[self.test_id, test_data]
        assert (
            names_train == names_test
        ), "Columns in the train and test data do not have the same ordering or contain different columns"
        return train_data, test_data, names_train

    def clean_columns(self, data):
        """Method to process (bring to cotinous representation, fix, clean remap) a dataset
        Args:
            data: Raw data as input, shape=(N,D)

        Returns:
            data: cleaned data, shape=(N,D)
        """
        ####### Juan's Part
        ex_mapping = {
            5: 0,
            1: 5,
            4: 1,
            2: 4,
            3: 2,
            4: 3,
            5: 4,
            7: 2,
            9: 2,
            -1: 2,
        }  # Maping 7, 9, and missing to intermediate value 2
        data = self.impute_mapping(data, ex_mapping, "GENHLTH")

        ex_mapping = {
            88: 0,
            77: 4,
            99: 4,
            -1: 4,
        }  # Mapping 77, 99, and missing values to mean of column
        data = self.impute_mapping(data, ex_mapping, "PHYSHLTH")

        ex_mapping = {
            88: 0,
            77: 5,
            99: 4.60,
            -1: 5,
        }  # Mapping 77, 99, and missing values to mean of column
        data = self.impute_mapping(data, ex_mapping, "MENTHLTH")

        ex_mapping = {
            88: 0,
            77: 3.03,
            99: 3.03,
            -1: 0,
        }  # Mapping 77, 99 to mean of column
        data = self.impute_mapping(data, ex_mapping, "POORHLTH")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "HLTHPLN1")

        ex_mapping = {3: 0, 2: 3, 1: 3, 7: 1, 9: 1}
        data = self.impute_mapping(data, ex_mapping, "PERSDOC2")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "MEDCOST")

        ex_mapping = {8: 0, 4: 5, 3: 4, 2: 3, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "CHECKUP1")

        ex_mapping = {3: 0, 1: 5, 7: 1, 2: 3, 4: 2, 5: 4, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "BPHIGH4")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "BPMEDS")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "BLOODCHO")

        ex_mapping = {4: 5, 3: 4, 2: 3, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "CHOLCHK")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "TOLDHI2")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "CVDSTRK3")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "ASTHMA3")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "ASTHNOW")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "CHCSCNCR")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "CHCOCNCR")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "CHCCOPD1")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "HAVARTH3")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "ADDEPEV2")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "CHCKIDNY")

        ex_mapping = {3: 0, 2: 3, 7: 2, 9: 2, -1: 2, 1: 5, 4: 1, 5: 4}
        data = self.impute_mapping(data, ex_mapping, "DIABETE3")

        ex_mapping = {98: 6, 99: 6, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "DIABAGE2")

        ex_mapping = {5: 0, 3: 5, 4: 3, 2: 4, 1: 2, 6: 1, 9: 0, -1: 2}
        data = self.impute_mapping(data, ex_mapping, "MARITAL")

        ex_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 4}
        data = self.impute_mapping(data, ex_mapping, "EDUCA")

        ex_mapping = {3: 0, 1: 3, 7: 0, 9: 0}
        data = self.impute_mapping(data, ex_mapping, "RENTHOM1")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "NUMHHOL2")

        ex_mapping = {7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "NUMPHON2")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "CPDEMO1")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "VETERAN3")

        ex_mapping = {6: 0, 2: 6, 3: 2, 4: 3, 5: 4, 1: 5, 8: 1, 9: 5}
        data = self.impute_mapping(data, ex_mapping, "EMPLOY1")

        ex_mapping = {88: 0, 99: 0.5, -1: 0.5}
        data = self.impute_mapping(data, ex_mapping, "CHILDREN")

        ex_mapping = {77: 0, 99: 5, -1: 5}
        data = self.impute_mapping(data, ex_mapping, "INCOME2")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "INTERNET")

        data["WEIGHT2"] = np.where(
            (data["WEIGHT2"] >= 9000) & (data["WEIGHT2"] <= 9998),
            (data["WEIGHT2"] - 9000) * 2.20462,
            data["WEIGHT2"],
        )
        ex_mapping = {7777: 184.25, 9999: 184.25, -1: 184.25}
        data = self.impute_mapping(data, ex_mapping, "WEIGHT2")

        data["HEIGHT3"] = np.where(
            (data["HEIGHT3"] >= 9000) & (data["HEIGHT3"] <= 9998),
            (data["HEIGHT3"] - 9000) * 3.28084,
            data["HEIGHT3"],
        )
        ex_mapping = {7777: 535.9, 9999: 535.9, -1: 535.9}
        data = self.impute_mapping(data, ex_mapping, "HEIGHT3")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "QSTLANG")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "PREGNANT")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "QLACTLM2")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "USEEQUIP")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "BLIND")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "DECIDE")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "DIFFWALK")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "DIFFDRES")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "DIFFALON")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "SMOKE100")

        ex_mapping = {3: 0, 1: 3, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "SMOKDAY2")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "STOPSMK2")

        ex_mapping = {
            8: 0,
            7: 8,
            6: 7,
            5: 6,
            4: 5,
            3: 4,
            2: 3,
            1: 2,
            77: 1,
            99: 1,
            -1: 0,
        }
        data = self.impute_mapping(data, ex_mapping, "LASTSMK2")

        ex_mapping = {7: 2, 9: 2, -1: 2}
        data = self.impute_mapping(data, ex_mapping, "USENOW3")

        data["ALCDAY5"] = np.where(
            (data["ALCDAY5"] >= 101) & (data["ALCDAY5"] <= 199),
            data["ALCDAY5"] - 100,
            data["ALCDAY5"],
        )
        data["ALCDAY5"] = np.where(
            (data["ALCDAY5"] >= 201) & (data["ALCDAY5"] <= 299),
            (data["ALCDAY5"] - 200) / 30,
            data["ALCDAY5"],
        )
        ex_mapping = {888: 0, 777: 0.43, 999: 0.43, -1: 0.43}
        data = self.impute_mapping(data, ex_mapping, "ALCDAY5")

        ex_mapping = {77: 1, 99: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "AVEDRNK2")

        ex_mapping = {77: 0, 88: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "DRNK3GE5")

        ex_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "MAXDRNKS")

        data["FRUITJU1"] = np.where(
            (data["FRUITJU1"] >= 101) & (data["FRUITJU1"] <= 199),
            data["FRUITJU1"] - 100,
            data["FRUITJU1"],
        )
        data["FRUITJU1"] = np.where(
            (data["FRUITJU1"] >= 201) & (data["FRUITJU1"] <= 299),
            (data["FRUITJU1"] - 200) / 7,
            data["FRUITJU1"],
        )
        data["FRUITJU1"] = np.where(
            (data["FRUITJU1"] >= 301) & (data["FRUITJU1"] <= 399),
            (data["FRUITJU1"] - 300) / 30,
            data["FRUITJU1"],
        )
        ex_mapping = {300: 1, 555: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "FRUITJU1")

        data["FRUIT1"] = np.where(
            (data["FRUIT1"] >= 101) & (data["FRUIT1"] <= 199),
            data["FRUIT1"] - 100,
            data["FRUIT1"],
        )
        data["FRUIT1"] = np.where(
            (data["FRUIT1"] >= 201) & (data["FRUIT1"] <= 299),
            (data["FRUIT1"] - 200) / 7,
            data["FRUIT1"],
        )
        data["FRUIT1"] = np.where(
            (data["FRUIT1"] >= 301) & (data["FRUIT1"] <= 399),
            (data["FRUIT1"] - 300) / 30,
            data["FRUIT1"],
        )
        ex_mapping = {300: 1, 555: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "FRUIT1")

        data["FVBEANS"] = np.where(
            (data["FVBEANS"] >= 101) & (data["FVBEANS"] <= 199),
            data["FVBEANS"] - 100,
            data["FVBEANS"],
        )
        data["FVBEANS"] = np.where(
            (data["FVBEANS"] >= 201) & (data["FVBEANS"] <= 299),
            (data["FVBEANS"] - 200) / 7,
            data["FVBEANS"],
        )
        data["FVBEANS"] = np.where(
            (data["FVBEANS"] >= 301) & (data["FVBEANS"] <= 399),
            (data["FVBEANS"] - 300) / 30,
            data["FVBEANS"],
        )
        ex_mapping = {300: 1, 555: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "FVBEANS")

        data["FVGREEN"] = np.where(
            (data["FVGREEN"] >= 101) & (data["FVGREEN"] <= 199),
            data["FVGREEN"] - 100,
            data["FVGREEN"],
        )
        data["FVGREEN"] = np.where(
            (data["FVGREEN"] >= 201) & (data["FVGREEN"] <= 299),
            (data["FVGREEN"] - 200) / 7,
            data["FVGREEN"],
        )
        data["FVGREEN"] = np.where(
            (data["FVGREEN"] >= 301) & (data["FVGREEN"] <= 399),
            (data["FVGREEN"] - 300) / 30,
            data["FVGREEN"],
        )
        ex_mapping = {300: 1, 555: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "FVGREEN")

        data["FVORANG"] = np.where(
            (data["FVORANG"] >= 101) & (data["FVORANG"] <= 199),
            data["FVORANG"] - 100,
            data["FVORANG"],
        )
        data["FVORANG"] = np.where(
            (data["FVORANG"] >= 201) & (data["FVORANG"] <= 299),
            (data["FVORANG"] - 200) / 7,
            data["FVORANG"],
        )
        data["FVORANG"] = np.where(
            (data["FVORANG"] >= 301) & (data["FVORANG"] <= 399),
            (data["FVORANG"] - 300) / 30,
            data["FVORANG"],
        )
        ex_mapping = {300: 1, 555: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "FVORANG")

        data["VEGETAB1"] = np.where(
            (data["VEGETAB1"] >= 101) & (data["VEGETAB1"] <= 199),
            data["VEGETAB1"] - 100,
            data["VEGETAB1"],
        )
        data["VEGETAB1"] = np.where(
            (data["VEGETAB1"] >= 201) & (data["VEGETAB1"] <= 299),
            (data["VEGETAB1"] - 200) / 7,
            data["VEGETAB1"],
        )
        data["VEGETAB1"] = np.where(
            (data["VEGETAB1"] >= 301) & (data["VEGETAB1"] <= 399),
            (data["VEGETAB1"] - 300) / 30,
            data["VEGETAB1"],
        )
        ex_mapping = {300: 1, 555: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "VEGETAB1")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "EXERANY2")

        ex_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "EXRACT11")

        data["EXEROFT1"] = np.where(
            (data["EXEROFT1"] >= 101) & (data["EXEROFT1"] <= 199),
            data["EXEROFT1"] - 100,
            data["EXEROFT1"],
        )
        data["EXEROFT1"] = np.where(
            (data["EXEROFT1"] >= 201) & (data["EXEROFT1"] <= 299),
            (data["EXEROFT1"] - 200) / 4,
            data["EXEROFT1"],
        )
        ex_mapping = {777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "EXEROFT1")

        ex_mapping = {777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "EXERHMM1")

        ex_mapping = {777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "EXERHMM2")

        ex_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "EXRACT21")

        data["EXEROFT2"] = np.where(
            (data["EXEROFT2"] >= 101) & (data["EXEROFT2"] <= 199),
            data["EXEROFT2"] - 100,
            data["EXEROFT2"],
        )
        data["EXEROFT2"] = np.where(
            (data["EXEROFT2"] >= 201) & (data["EXEROFT2"] <= 299),
            (data["EXEROFT2"] - 200) / 4,
            data["EXEROFT2"],
        )
        ex_mapping = {777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "EXEROFT2")

        data["STRENGTH"] = np.where(
            (data["STRENGTH"] >= 101) & (data["STRENGTH"] <= 199),
            data["STRENGTH"] - 100,
            data["STRENGTH"],
        )
        data["STRENGTH"] = np.where(
            (data["STRENGTH"] >= 201) & (data["STRENGTH"] <= 299),
            (data["STRENGTH"] - 200) / 4,
            data["STRENGTH"],
        )
        ex_mapping = {777: 0, 888: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "STRENGTH")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "LMTJOIN3")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "ARTHDIS2")

        ex_mapping = {3: 0, 1: 3, 2: 1, 3: 2, 7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "ARTHSOCL")

        ex_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "JOINPAIN")

        ex_mapping = {1: 6, 5: 1, 2: 5, 7: 2, 9: 2, -1: 2, 8: 0, 3: 10, 4: 3, 10: 4}
        data = self.impute_mapping(data, ex_mapping, "SEATBELT")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "FLUSHOT6")

        ex_mapping = {777777: 0, 999999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "FLSHTMY2")

        ex_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "IMFVPLAC")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "PNEUVAC3")

        ex_mapping = {777777: 0, 999999: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "HIVTST6")

        ex_mapping = {2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, ex_mapping, "HIVTSTD3")

        ex_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, ex_mapping, "WHRTST10")

        ##PREDIABETES
        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "PDIABTST")

        new_mapping = {3: 0, 7: 1, 9: 1, -1: 1, 1: 3}  # 2 remains 2
        data = self.impute_mapping(data, new_mapping, "PREDIAB1")

        # DIABETES

        new_mapping = {2: 0, 1: 2, 9: 1, -1: 0}
        data = self.impute_mapping(data, new_mapping, "INSULIN")

        data["BLDSUGAR"] = np.where(
            (data["BLDSUGAR"] >= 101) & (data["BLDSUGAR"] <= 199),
            data["BLDSUGAR"] - 100,
            data["BLDSUGAR"],
        )
        data["BLDSUGAR"] = np.where(
            (data["BLDSUGAR"] >= 201) & (data["BLDSUGAR"] <= 299),
            (data["BLDSUGAR"] - 200) / 7,
            data["BLDSUGAR"],
        )
        data["BLDSUGAR"] = np.where(
            (data["BLDSUGAR"] >= 301) & (data["BLDSUGAR"] <= 399),
            (data["BLDSUGAR"] - 300) / 30,
            data["BLDSUGAR"],
        )
        data["BLDSUGAR"] = np.where(
            (data["BLDSUGAR"] >= 401) & (data["BLDSUGAR"] <= 499),
            (data["BLDSUGAR"] - 400) / 365,
            data["BLDSUGAR"],
        )
        new_mapping = {888: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "BLDSUGAR")

        data["FEETCHK2"] = np.where(
            (data["FEETCHK2"] >= 101) & (data["FEETCHK2"] <= 199),
            data["FEETCHK2"] - 100,
            data["FEETCHK2"],
        )
        data["FEETCHK2"] = np.where(
            (data["FEETCHK2"] >= 201) & (data["FEETCHK2"] <= 299),
            (data["FEETCHK2"] - 200) / 7,
            data["FEETCHK2"],
        )
        data["FEETCHK2"] = np.where(
            (data["FEETCHK2"] >= 301) & (data["FEETCHK2"] <= 399),
            (data["FEETCHK2"] - 300) / 30,
            data["FEETCHK2"],
        )
        data["FEETCHK2"] = np.where(
            (data["FEETCHK2"] >= 401) & (data["FEETCHK2"] <= 499),
            (data["FEETCHK2"] - 400) / 365,
            data["FEETCHK2"],
        )
        new_mapping = {555: 0, 888: 0, 777: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "FEETCHK2")

        new_mapping = {88: 0, 77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "DOCTDIAB")

        new_mapping = {88: 0, 77: 0, 98: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "CHKHEMO3")

        new_mapping = {88: 0, 77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "FEETCHK")

        new_mapping = {8: 0, 7: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}  # 3 remains 3
        data = self.impute_mapping(data, new_mapping, "EYEEXAM")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "DIABEYE")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "DIABEDU")

        # CAREGIVER

        new_mapping = {8: 0, 2: 0, 1: 2, 7: 1, 9: 1, -1: 1}
        data = self.impute_mapping(data, new_mapping, "CAREGIV1")

        new_mapping = {77: 0, 99: 0, -1: 0}  # SOME OVERLAPS WITH CAREGIV1
        data = self.impute_mapping(data, new_mapping, "CRGVREL1")

        new_mapping = {7: 0, 9: 0, -1: 0}  # SOME OVERLAPS WITH CAREGIV1
        data = self.impute_mapping(data, new_mapping, "CRGVLNG1")

        new_mapping = {7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "CRGVHRS1")

        new_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "CRGVPRB1")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CRGVPERS")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CRGVHOUS")

        new_mapping = {6: 0, 7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "CRGVMST2")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CRGVEXPT")

        # VISUAL IMPAIRMENT

        new_mapping = {1: 0, 7: 1, -1: 1, 5: 5, 6: 5, 8: 5}
        data = self.impute_mapping(data, new_mapping, "VIDFCLT2")

        new_mapping = {1: 0, 7: 1, -1: 1, 5: 5, 6: 5}
        data = self.impute_mapping(data, new_mapping, "VIREDIF3")

        new_mapping = {5: 0, 7: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "VIPRFVS2")

        new_mapping = {8: 0, 77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "VINOCRE2")

        new_mapping = {5: 0, 8: 0, 7: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "VIEYEXM2")

        new_mapping = {2: 0, 8: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "VIINSUR2")

        new_mapping = {3: 0, 2: 1, 7: 2, -1: 2, 1: 3}
        data = self.impute_mapping(data, new_mapping, "VICTRCT4")

        new_mapping = {2: 0, 7: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "VIGLUMA2")

        new_mapping = {2: 0, 7: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "VIMACDG2")

        # COGNITIVE DECLINE

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CIMEMLOS")

        new_mapping = {5: 0, 7: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "CDHOUSE")

        new_mapping = {5: 0, 7: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "CDASSIST")

        new_mapping = {5: 0, 7: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "CDHELP")

        new_mapping = {5: 0, 7: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "CDSOCIAL")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CDDISCUS")

        # SODIUM/SALT RELATED BEHAVIOUR

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "WTCHSALT")

        data["LONGWTCH"] = np.where(
            (data["LONGWTCH"] >= 101) & (data["LONGWTCH"] <= 199),
            data["LONGWTCH"] - 100,
            data["LONGWTCH"],
        )
        data["LONGWTCH"] = np.where(
            (data["LONGWTCH"] >= 201) & (data["LONGWTCH"] <= 299),
            (data["LONGWTCH"] - 200) * 7,
            data["LONGWTCH"],
        )
        data["LONGWTCH"] = np.where(
            (data["LONGWTCH"] >= 301) & (data["LONGWTCH"] <= 399),
            (data["LONGWTCH"] - 300) * 30,
            data["LONGWTCH"],
        )
        data["LONGWTCH"] = np.where(
            (data["LONGWTCH"] >= 401) & (data["LONGWTCH"] <= 499),
            (data["LONGWTCH"] - 400) * 365,
            data["LONGWTCH"],
        )
        new_mapping = {777: 0, 999: 0, -1: 0, 555: np.max(data["LONGWTCH"])}
        data = self.impute_mapping(data, new_mapping, "LONGWTCH")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "DRADVISE")

        ##ADULT ASTHMA HISTORY

        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 11) & (data["ASTHMAGE"] <= 21), 2, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 22) & (data["ASTHMAGE"] <= 32), 3, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 33) & (data["ASTHMAGE"] <= 43), 4, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 44) & (data["ASTHMAGE"] <= 54), 5, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 55) & (data["ASTHMAGE"] <= 65), 6, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 66) & (data["ASTHMAGE"] <= 76), 7, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 77) & (data["ASTHMAGE"] <= 87), 8, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where(
            (data["ASTHMAGE"] >= 88) & (data["ASTHMAGE"] <= 95), 9, data["ASTHMAGE"]
        )
        data["ASTHMAGE"] = np.where((data["ASTHMAGE"] == 96), 10, data["ASTHMAGE"])
        new_mapping = {98: 0, 99: 0, -1: 0, 97: 1}
        data = self.impute_mapping(data, new_mapping, "ASTHMAGE")

        new_mapping = {2: 0, 7: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "ASATTACK")

        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 1) & (data["ASERVIST"] <= 10), 2, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 11) & (data["ASERVIST"] <= 21), 3, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 22) & (data["ASERVIST"] <= 32), 4, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 33) & (data["ASERVIST"] <= 43), 5, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 44) & (data["ASERVIST"] <= 54), 6, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 55) & (data["ASERVIST"] <= 65), 7, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 66) & (data["ASERVIST"] <= 76), 8, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where(
            (data["ASERVIST"] >= 77) & (data["ASERVIST"] <= 86), 9, data["ASERVIST"]
        )
        data["ASERVIST"] = np.where((data["ASERVIST"] == 87), 10, data["ASERVIST"])
        new_mapping = {88: 0, 98: 1, 99: 1, -1: 1}
        data = self.impute_mapping(data, new_mapping, "ASERVIST")

        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 1) & (data["ASDRVIST"] <= 10), 2, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 11) & (data["ASDRVIST"] <= 21), 3, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 22) & (data["ASDRVIST"] <= 32), 4, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 33) & (data["ASDRVIST"] <= 43), 5, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 44) & (data["ASDRVIST"] <= 54), 6, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 55) & (data["ASDRVIST"] <= 65), 7, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 66) & (data["ASDRVIST"] <= 76), 8, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where(
            (data["ASDRVIST"] >= 77) & (data["ASDRVIST"] <= 86), 9, data["ASDRVIST"]
        )
        data["ASDRVIST"] = np.where((data["ASDRVIST"] == 87), 10, data["ASDRVIST"])
        new_mapping = {88: 0, 98: 1, -1: 1}
        data = self.impute_mapping(data, new_mapping, "ASDRVIST")

        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 1) & (data["ASRCHKUP"] <= 10), 2, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 11) & (data["ASRCHKUP"] <= 21), 3, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 22) & (data["ASRCHKUP"] <= 32), 4, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 33) & (data["ASRCHKUP"] <= 43), 5, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 44) & (data["ASRCHKUP"] <= 54), 6, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 55) & (data["ASRCHKUP"] <= 65), 7, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 66) & (data["ASRCHKUP"] <= 76), 8, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where(
            (data["ASRCHKUP"] >= 77) & (data["ASRCHKUP"] <= 86), 9, data["ASRCHKUP"]
        )
        data["ASRCHKUP"] = np.where((data["ASRCHKUP"] == 87), 10, data["ASRCHKUP"])
        new_mapping = {88: 0, 98: 1, 99: 1, -1: 1}
        data = self.impute_mapping(data, new_mapping, "ASRCHKUP")

        new_mapping = {777: 0, 888: 0, 999: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "ASACTLIM")

        new_mapping = {8: 0, 7: 1, 9: 1, -1: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
        data = self.impute_mapping(data, new_mapping, "ASYMPTOM")

        new_mapping = {8: 0, 7: 1, -1: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
        data = self.impute_mapping(data, new_mapping, "ASNOSLEP")

        new_mapping = {8: 0, 7: 1, 9: 1, -1: 1, 1: 2, 2: 3, 3: 4}
        data = self.impute_mapping(data, new_mapping, "ASTHMED3")

        new_mapping = {8: 0, 7: 1, -1: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
        data = self.impute_mapping(data, new_mapping, "ASINHALR")

        # CV HEALTH

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "HAREHAB1")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "STREHAB1")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CVDASPRN")

        new_mapping = {3: 0, 2: 1, 7: 2, -1: 2, 1: 3}
        data = self.impute_mapping(data, new_mapping, "ASPUNSAF")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "RLIVPAIN")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "RDUCHART")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "RDUCSTRK")

        # ARTHRITIS MGMT

        new_mapping = {7: 0, 9: 0, -1: 0, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "ARTTODAY")

        # SUGGESTIONS FOR LOSING WEIGHT/PHY ACT: MAYBE DELETED

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "ARTHWGT")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "ARTHEXER")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "ARTHEDU")

        # TDP

        new_mapping = {4: 0, 7: 1, 9: 1, -1: 1, 3: 2, 2: 3, 1: 4}
        data = self.impute_mapping(data, new_mapping, "TETANUS")

        new_mapping = {2: 0, 3: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "HPVADVC2")

        new_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "HPVADSHT")

        # SHINGLES

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "SHINGLE2")

        # BREAST AND CERVICAL CANCER

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "HADMAM")

        new_mapping = {7: 0, 9: 0, -1: 0, 5: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "HOWLONG")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "HADPAP2")

        new_mapping = {7: 0, 9: 0, -1: 0, 5: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "LASTPAP2")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "HPVTEST")

        new_mapping = {7: 0, 9: 0, -1: 0, 5: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "HPLSTTST")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "HADHYST2")

        # CLINICAL BREAST EXAM

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "PROFEXAM")

        new_mapping = {7: 0, 9: 0, -1: 0, 5: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "LENGEXAM")

        # COLORECTAL CANCER SCREENING

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "BLDSTOOL")

        new_mapping = {7: 0, 9: 0, -1: 0, 5: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "LSTBLDS3")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "HADSIGM3")

        new_mapping = {7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "HADSGCO1")

        new_mapping = {7: 0, 9: 0, -1: 0, 6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6}
        data = self.impute_mapping(data, new_mapping, "LASTSIG3")

        # PROSTRATE CANCER SCREENING

        # TALKING ABOUT ADV OF PSA TEST: MAYBE DELETED
        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "PCPSAAD2")

        # TALKING ABOUT DISADV OF PSA TEST: MAYBE DELETED
        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "PCPSADI1")

        # TALKING ABOUT RECOMMENDING PSA TEST: MAYBE DELETED
        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "PCPSARE1")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "PSATEST1")

        new_mapping = {7: 0, 9: 0, -1: 0, 5: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "PSATIME")

        new_mapping = {7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "PCPSARS1")

        # WHO MADE DECISION FOR PSA TEST: MAYBE DELETED
        new_mapping = {4: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "PCPSADE1")

        new_mapping = {-1: 0, 1: 5, 7: 1, 9: 0}
        data = self.impute_mapping(data, new_mapping, "PCDMDECN")

        # SOCIAL CONTEXT

        new_mapping = {5: 0, 7: 1, 8: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "SCNTMNY1")

        new_mapping = {5: 0, 7: 1, 8: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "SCNTMEL1")

        new_mapping = {7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "SCNTPAID")

        data["SCNTWRK1"] = np.where(
            (data["SCNTWRK1"] == 98) | (data["SCNTWRK1"] == -1), 0, data["SCNTWRK1"]
        )
        new_mapping = {97: data["SCNTWRK1"].mean(), 99: data["SCNTWRK1"].mean()}
        data = self.impute_mapping(data, new_mapping, "SCNTWRK1")

        new_mapping = {7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "SCNTLPAD")

        data["SCNTLWK1"] = np.where(
            (data["SCNTLWK1"] == 98) | (data["SCNTLWK1"] == -1), 0, data["SCNTLWK1"]
        )
        new_mapping = {97: data["SCNTLWK1"].mean(), 99: data["SCNTLWK1"].mean()}
        data = self.impute_mapping(data, new_mapping, "SCNTLWK1")

        # SEXUAL ORIENTATION

        new_mapping = {7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "SXORIENT")

        new_mapping = {4: 0, 9: 0, -1: 0, 1: 2, 3: 2, 7: 1, -1: 0}
        data = self.impute_mapping(data, new_mapping, "TRNSGNDR")

        # GENDER OF CHILD

        new_mapping = {9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "RCSGENDR")

        new_mapping = {6: 0, 7: 0, 9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "RCSRLTN2")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CASTHDX2")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "CASTHNO2")

        # EM SUPPORT

        new_mapping = {5: 0, 7: 1, 8: 1, 9: 1, -1: 1, 4: 2, 2: 4, 1: 5}
        data = self.impute_mapping(data, new_mapping, "EMTSUPRT")

        new_mapping = {4: 0, 3: 1, 7: 2, 9: 2, -1: 2, 2: 3, 1: 4}
        data = self.impute_mapping(data, new_mapping, "LSATISFY")

        # ANXIETY AND DEP

        # just imputing 88, means for 77, 99 and -1 will be filled at the end
        data["ADPLEASR"] = np.where((data["ADPLEASR"] == 88), 0, data["ADPLEASR"])
        data["ADDOWN"] = np.where((data["ADDOWN"] == 88), 0, data["ADDOWN"])
        data["ADSLEEP"] = np.where((data["ADSLEEP"] == 88), 0, data["ADSLEEP"])
        data["ADENERGY"] = np.where((data["ADENERGY"] == 88), 0, data["ADENERGY"])
        data["ADEAT1"] = np.where((data["ADEAT1"] == 88), 0, data["ADEAT1"])
        data["ADFAIL"] = np.where((data["ADFAIL"] == 88), 0, data["ADFAIL"])
        data["ADTHINK"] = np.where((data["ADTHINK"] == 88), 0, data["ADTHINK"])
        data["ADMOVE"] = np.where((data["ADMOVE"] == 88), 0, data["ADMOVE"])

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "MISTMNT")

        new_mapping = {2: 0, 7: 1, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "ADANXEV")

        # WEIGHTING VARIABLES

        new_mapping = {5: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "MSCODE")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_STSTR")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_STRWT")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_RAWRAKE")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_WT2RAKE")

        new_mapping = {9: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "_DUALUSE")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_DUALCOR")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_LLCPWT")

        # CHILD DEMO VAR

        new_mapping = {2: 0, 9: 1, -1: 1, 1: 2}
        data = self.impute_mapping(data, new_mapping, "_CHISPNC")

        new_mapping = {77: 0, 99: 0, -1: 0}
        data = self.impute_mapping(data, new_mapping, "_CRACE1")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_CPRACE")

        new_mapping = {-1: 0}
        data = self.impute_mapping(data, new_mapping, "_CLLCPWT")

        temp_list = [
            "_LTASTH1",
            "_RFCHOL",  # has one category more than the other two, but mapping still works
            "_CASTHM1",
        ]

        for col in temp_list:
            ex_mapping = {
                1: 0,
                2: 1,
            }
            data = self.impute_mapping(data, ex_mapping, col)

        data = self.impute_mapping(
            data,
            {
                3: 0,
                2: 1,
                1: 2,
            },
            "_ASTHMS1",
        )

        data = self.impute_mapping(
            data,
            {
                2: 0,
            },
            "_DRDXAR1",
        )

        data = self.impute_mapping(data, {4: 0, 3: 1, 2: 2, 1: 3}, "_SMOKER3")

        data = self.impute_mapping(
            data,
            {
                1: 0,
                2: 1,
            },
            "_RFSMOK3",
        )

        data = self.impute_mapping(
            data,
            {
                1: 1,
                2: 0,
            },
            "DRNKANY5",
        )

        # leisrue time phyiscal activtiy yes ir bi
        data = self.impute_mapping(data, {2: 0, 1: 1}, "_TOTINDA")

        data = self.impute_mapping(
            data,
            {
                2: 1,
                1: 0,
            },
            "_PAINDX1",
        )

        temp_cols = [
            "_PA150R2",
            "_PA300R2",
        ]

        for col in temp_cols:
            data = self.impute_mapping(
                data,
                {
                    3: 0,
                    2: 1,
                    1: 3,
                },
                col,
            )

        temp_cols = ["_PA30021", "_PA30021", "_PASTAE1", "_RFSEAT2", "_RFSEAT3"]
        for col in temp_cols:
            data = self.impute_mapping(
                data,
                {
                    2: 0,
                    1: 1,
                },
                col,
            )

        return data

    def impute_means(
        self,
    ):
        """Method to impute means for selected columns and values for the processed dataset (changes in place)"""
        grouping = {
            (9,): [
                "_RFHLTH",
                "_HCVU651",
                "_RFHYPE5",
                "_CHOLCHK",  # 1 is check in 5 years, 2 is check before 5 years, one is never a check
                "_RFBMI5",  # BMI in categories overweight, obese
                "_CHLDCNT",  # childcount
                "_EDUCAG",
                "_INCOMG",
                "_PACAT1",  # acitivty vateogries
                "_LTASTH1",
                "_RFCHOL",  # has one category more than the other two, but mapping still works
                "_CASTHM1",
                "_DRDXAR1",
                "_ASTHMS1",
                "_SMOKER3",
                "_RFSMOK3",
                "DRNKANY5",
                "_TOTINDA",
                "_PAINDX1",
                "_PA30021",
                "_PA30021",
                "_PASTAE1",
                "_RFSEAT2",
                "_RFSEAT3",
                "_PA150R2",
                "_PA300R2",
                "_HISPANC",
                "_RACE",
                "_RACEG21",
                "_RACEGR3",
            ],
            (
                9,
                -1,
            ): ["_LMTACT1", "_LMTWRK1", "_LMTSCL1", "_FLSHOT6", "_PNEUMO2", "_AIDTST3"],
            (-1,): [
                "_DRDXAR1",
                "_LTASTH1",
                "_RFCHOL",  # has one category more than the other two, but mapping still works
                "_CASTHM1",
                "HTIN4",  # height in inches
                "HTM4",  # height in meters
                "WTKG3",  # weight in kg
                "_BMI5",  # calculated BMI
                "_BMI5CAT",  # ceategorized BMI
                "FTJUDA1_",
                "FRUTDA1_",
                "BEANDAY_",
                "GRENDAY_",
                "ORNGDAY_",
                "VEGEDA1_",
                "_FRUTSUM",  # total fruits per day
                "_VEGESUM",
                "METVL11_",
                "METVL21_",
                "ACTIN11_",  # Intnesity of activity 1
                "ACTIN21_",  # intensity of activity 2
                "PADUR1_",  # minutes of activity 1
                "PADUR2_",  # minutes of activity 2
                "PAFREQ1_",  # frquency of activity 1
                "PAFREQ2_",  # frquency of activity 2
                "_MINAC11",  # minutes of pyhsical activity 1
                "_MINAC21",  # minutes of pyhsical activity 2
                "STRFREQ_",  # frequency of strength training
                "PAMIN11_",  # minutes of pyhsical activity 1 -- DELETE? DUPLICATE???
                "PAMIN21_",  # minutes of pyhsical activity 2 -- DELETE? DUPLICATE???
                "PA1MIN_",  # total minutes of pyhsical activity per week -- TODO: ONLY KEEP THIS???
                "PA1VIGM_",  # total number of vigouorus minutes
                "PAVIG11_",  # number of vigourous minutes per week for activity 1
                "PAVIG21_",  # number of vigourous minutes per week for activity 2
                "_PAREC1",  # how many guidelines are met
                "_RACE_G1",
            ],
            (14,): ["_AGEG5YR"],
            (3,): ["_AGE65YR"],
            (7,): ["DRNKANY5"],
            (900,): ["DRNKANY5", "DROCDY3_"],
            (99900,): [
                "_DRNKWEK",
                "MAXVO2_",
                "FC60_",
                "PAFREQ1_",  # frquency of activity 1
                "PAFREQ2_",  # frquency of activity 2
                "STRFREQ_",  # frequency of strength trainin
            ],
            (99, 77): [
                "_PRACE1",
                "_MRACE1",
            ],
            (77, 99, -1): [
                "ADPLEASR",
                "ADDOWN",
                "ADSLEEP",
                "ADENERGY",
                "ADEAT1",
                "ADFAIL",
                "ADTHINK",
                "ADMOVE",
            ],
        }

        # calculate training means while exluding any value that will be replaced in that column
        for from_tuple, columns_list in grouping.items():
            for col in columns_list:
                # filter out any data that will be needed before calculating the mean
                mask = [x not in from_tuple for x in self.train_data[col]]
                self.training_means[col] = np.mean(self.train_data[col][mask])

        # apply to training data
        for from_tuple, columns_list in grouping.items():
            for col in columns_list:
                for from_val in from_tuple:
                    self.train_processed[col] = np.where(
                        self.train_processed[col] == from_val,
                        self.training_means[col],
                        self.train_processed[col],
                    )

        # apply to test data
        for from_tuple, columns_list in grouping.items():
            for col in columns_list:
                for from_val in from_tuple:
                    self.test_processed[col] = np.where(
                        self.test_processed[col] == from_val,
                        self.training_means[col],
                        self.test_processed[col],
                    )

    def subsetted_data(
        self,
    ):
        """Method to create and access the subsetted data (removal of seemingly duplicated columns)

        Returns:
            train_data: subsetted training data, shape=(N,D)
            test_data: subsetted testing data, shape=(N,D)
            names_train: columns names from training dataset (list)
        """
        DELETE = [
            # delte these age columns, since we have the actual age in Age80??
            "_AGEG5Y",
            "_AGE65YR",
            "_AGE_G",
            # categorized BMI, since we have actual BMI
            "_RFBMI5",
            "_BMI5CAT",
            # Number of missing fruit responses - delete as it seems pretty useless
            "_MISFRTN",
            "_MISVEGN",
            "_FRTRESP",
            "_VEGRESP"
            # consumes more than x fruits or vegetables per day: uselss and highly skewed
            "_FRTLT1",
            "_VEGLT1",
            "_FRT16",
            "_VEG23",
            # if any missing values in fruits or vegetables excluded from analyiss
            "_FRUITEX",
            "_VEGETEX",
            # if missing phyiscal activity data
            "PAMISS1_",
            "PAMIN11_",  # minutes of pyhsical activity 1
            "PAMIN21_",  # minutes of pyhsical activity 2
            # wearing seatbelt is irrelevant
            "_RFSEAT2",
            "_RFSEAT3",
            # Cols from Anand's Part
            "TRNSGNDR",
            "PCDMDECN",
            "VINOCRE2",
            "CRGVMST2",
            "CRGVREL1",
            "_STSTR",
            "_STRWT",
            "_RAWRAKE",
            "_WT2RAKE",
            "_DUALUSE",
            "_DUALCOR",
            "_LLCPWT",
            "_CPRACE",
            "_CLLCPWT",
            "MSCODE",
            # cols from Juan's part
            "NUMHHOL2",
            "NUMPHON2",
            "CPDEMO1",
            "QSTVER",  # irrelevant
            "QSTLANG",  # irrelevant
            "HIVTSTD3",
            "FLSHTMY2",
            "HIVTST6",  # already captured by _AIDTST3
            "SEATBELT",  # already captured by _RFSEAT3
            "LMTJOIN3",  # already captured by _LMTACT1
            "ARTHDIS2",  # already captured by _LMTWRK1
            "ARTHSOCL",  # already captured by _LMTSCL1
            "EXERANY2",  # valued implied by PA1MIN_ and _TOTINDA
            "EXEROFT1",  # already captured by PAVIG11_
            "EXERHMM1",  # (cannot interpret values) already captured by
            "EXEROFT2",  # already captured by PAVIG21_
            "EXERHMM2",  # (cannot interpret values) already captured by
            "STRENGTH",  # already captured by STRFREQ_
            "FRUITJU1",  # already captured by FTJUDA1_
            "FRUIT1",  # already captured by FRUTDA1_
            "FVBEANS",  # already captured by BEANDAY_
            "FVGREEN",  # already captured by GRENDAY_
            "FVORANG",  # already captured by ORNGDAY_
            "VEGETAB1",  # already captured by VEGEDA1_
            "ALCDAY5",  # already captured by DROCDY3_
            "HAVARTH3",  # already captured by _DRDXAR1
            "ASTHMA3",  # already captured by _ASTHMS1
            "ASTHNOW",  # already captured by _ASTHMS1
            "CHOLCHK",  # already captured by _CHOLCHK
            "TOLDHI2",  # already captured by _RFCHOL
            "BLOODCHO",  # already captured by _RFCHOL
            "HLTHPLN1",  # already captured by _HCVU651
            "FLUSHOT6",  # already captured by _FLSHOT6
            "PNEUVAC3",  # already captured by _PNEUMO2
            "CHILDREN",  # already captured by _CHLDCNT
            "WEIGHT2",  # already captured by WTKG3
            "HEIGHT3",  # already captured by HTM4 and HTIN4
            "_RFHYPE5",  # already captured by BPHIGH4
            "_RFHLTH",  # already captured by GENHLTH
            "_INCOMG",  # more detail in INCOME2
            "_EDUCAG",  # more detail in EDUCA
            # Information on Race
            "_PRACE1",
            "_MRACE1",
            "_HISPANC",
            "_RACE",
            "_RACEG21",
            "_RACEGR3",
            "_RACE_G1",
        ]

        # modify a copy
        train_data = deepcopy(self.train_processed)
        test_data = deepcopy(self.test_processed)

        self.train_subsetted = self.remove_field_name(train_data, *DELETE)
        self.test_subsetted = self.remove_field_name(test_data, *DELETE)

        # convert to array
        names_train, train_data = self.convert_to_array(self.train_subsetted)
        names_test, test_data = self.convert_to_array(self.test_subsetted)
        # add ID back
        train_data = np.c_[self.train_id, train_data]
        test_data = np.c_[self.test_id, test_data]
        assert (
            names_train == names_test
        ), "Columns in the train and test data do not have the same ordering or contain different columns"
        return train_data, test_data, names_train

    def curated_data(
        self,
    ):
        """Method to create and access the data containing the top 200 features by a MI score

        Returns:
            train_data: selected training data, shape=(N,D)
            test_data: selected testing data, shape=(N,D)
            names_train: columns names from training dataset (list)
        """
        # Top 200 features according to MI Score
        CURATED = [
            "_AGE80",
            "_AGEG5YR",
            "_AGE_G",
            "EMPLOY1",
            "MAXVO2_",
            "BPMEDS",
            "BPHIGH4",
            "_RFHYPE5",
            "_RFHLTH",
            "GENHLTH",
            "_AGE65YR",
            "_HCVU651",
            "_RFCHOL",
            "DIFFWALK",
            "TOLDHI2",
            "_PNEUMO2",
            "_FLSHOT6",
            "_LMTSCL1",
            "QLACTLM2",
            "PHYSHLTH",
            "PNEUVAC3",
            "_LMTACT1",
            "_LMTWRK1",
            "USEEQUIP",
            "CVDSTRK3",
            "DIABETE3",
            "HAVARTH3",
            "_DRDXAR1",
            "DIABAGE2",
            "CHCCOPD1",
            "ARTHSOCL",
            "JOINPAIN",
            "CHOLCHK",
            "INTERNET",
            "LMTJOIN3",
            "POORHLTH",
            "MARITAL",
            "_CHLDCNT",
            "CHILDREN",
            "DIFFALON",
            "EXRACT11",
            "CHCKIDNY",
            "INCOME2",
            "_CHOLCHK",
            "ACTIN11_",
            "VETERAN3",
            "_INCOMG",
            "IMFVPLAC",
            "_SMOKER3",
            "EXRACT21",
            "ARTHDIS2",
            "METVL11_",
            "SMOKE100",
            "LASTSMK2",
            "RCSRLTN2",
            "BLOODCHO",
            "DOCTDIAB",
            "DIFFDRES",
            "CHKHEMO3",
            "FEETCHK",
            "DROCDY3_",
            "BLDSUGAR",
            "ACTIN21_",
            "_CPRACE",
            "MAXDRNKS",
            "_CRACE1",
            "PADUR1_",
            "EYEEXAM",
            "AVEDRNK2",
            "METVL21_",
            "EXERHMM1",
            "BLIND",
            "DECIDE",
            "CHECKUP1",
            "DRNKANY5",
            "FEETCHK2",
            "RCSGENDR",
            "EDUCA",
            "PERSDOC2",
            "_EDUCAG",
            "_CHISPNC",
            "PADUR2_",
            "INSULIN",
            "EXERHMM2",
            "HAREHAB1",
            "DIABEYE",
            "ALCDAY5",
            "QSTVER",
            "CASTHDX2",
            "_PACAT1",
            "CHCOCNCR",
            "CHCSCNCR",
            "EXEROFT1",
            "_PA300R2",
            "_PA150R2",
            "_TOTINDA",
            "EXEROFT2",
            "SEX",
            "_BMI5",
            "FLUSHOT6",
            "_STSTR",
            "_BMI5CAT",
            "CDASSIST",
            "DRNK3GE5",
            "WTKG3",
            "_CLLCPWT",
            "CDHOUSE",
            "ADDEPEV2",
            "MENTHLTH",
            "_RFBING5",
            "DIABEDU",
            "EXERANY2",
            "CDSOCIAL",
            "SCNTLWK1",
            "STRENGTH",
            "SCNTWRK1",
            "WHRTST10",
            "FLSHTMY2",
            "PAFREQ1_",
            "_RFBMI5",
            "STRFREQ_",
            "SCNTPAID",
            "WEIGHT2",
            "_LLCPWT",
            "PAFREQ2_",
            "_STRWT",
            "_PAREC1",
            "_CASTHM1",
            "_WT2RAKE",
            "ASTHNOW",
            "SCNTLPAD",
            "ARTTODAY",
            "HIVTSTD3",
            "_DUALCOR",
            "_LTASTH1",
            "ASTHMA3",
            "_ASTHMS1",
            "_RACE",
            "HIVTST6",
            "HTM4",
            "HTIN4",
            "CDDISCUS",
            "_PASTRNG",
            "_AIDTST3",
            "HLTHPLN1",
            "_PASTAE1",
            "LONGWTCH",
            "CDHELP",
            "ARTHEDU",
            "RLIVPAIN",
            "LASTSIG3",
            "MSCODE",
            "HEIGHT3",
            "HADSGCO1",
            "FRUTDA1_",
            "_PRACE1",
            "ASPUNSAF",
            "CIMEMLOS",
            "FRUIT1",
            "HPVADVC2",
            "_MRACE1",
            "CRGVREL1",
            "PA1VIGM_",
            "_RACE_G1",
            "_RACEGR3",
            "_VEGESUM",
            "_DRNKWEK",
            "ADENERGY",
            "GRENDAY_",
            "ARTHWGT",
            "FRUITJU1",
            "FTJUDA1_",
            "FVGREEN",
            "LSTBLDS3",
            "SHINGLE2",
            "_DUALUSE",
            "_VEGLT1",
            "ADSLEEP",
            "CPDEMO1",
            "ADMOVE",
            "ARTHEXER",
            "CVDASPRN",
            "_HISPANC",
            "PDIABTST",
            "PAVIG11_",
            "SEATBELT",
            "PREGNANT",
            "HADHYST2",
            "_RFDRHV5",
            "ADEAT1",
            "_RACEG21",
            "ADDOWN",
            "HADMAM",
            "ADFAIL",
            "PREDIAB1",
            "PAVIG21_",
            "ADPLEASR",
            "ADTHINK",
            "VEGETAB1",
            "VEGEDA1_",
        ]
        to_remove = [
            name for name in self.train_processed.dtype.names if name not in CURATED
        ]

        # modify a copy
        train_data = deepcopy(self.train_processed)
        test_data = deepcopy(self.test_processed)
        self.train_curated = self.remove_field_name(train_data, *to_remove)
        self.test_curated = self.remove_field_name(test_data, *to_remove)

        # convert to array
        names_train, train_data = self.convert_to_array(self.train_curated)
        names_test, test_data = self.convert_to_array(self.test_curated)
        # add ID back
        train_data = np.c_[self.train_id, train_data]
        test_data = np.c_[self.test_id, test_data]
        assert (
            names_train == names_test
        ), "Columns in the train and test data do not have the same ordering or contain different columns"
        return train_data, test_data, names_train


def main():
    DATA_PATH = r"C:\data_repository\ML\raw_data"  # path to your data
    train_path = os.path.join(DATA_PATH, "x_train.csv")
    test_path = os.path.join(DATA_PATH, "x_test.csv")
    train_label_path = os.path.join(DATA_PATH, "y_train.csv")

    data = Data(train_path, test_path, train_label_path)  # read in data

    train_labels = data.train_labels

    (
        train_raw,
        test_raw,
        raw_cols,
    ) = (
        data.raw_data()
    )  # Access Dataset 1: only NAs filled with -1 and record IDs deleted

    (
        train_processed,
        test_processed,
        processed_cols,
    ) = (
        data.process_data()
    )  # Access Dataset 2: processed data with reformatted columns, NAs handled properly etc.

    # top 100 feature indices discovered by Lasso (by setting lambda1 = 5)
    lasso_feature_indices = np.array(
        [
            24,
            12,
            0,
            222,
            220,
            13,
            129,
            204,
            18,
            9,
            77,
            21,
            22,
            131,
            223,
            206,
            8,
            46,
            43,
            221,
            31,
            39,
            233,
            6,
            207,
            235,
            232,
            25,
            41,
            20,
            83,
            35,
            5,
            19,
            1,
            16,
            72,
            118,
            212,
            54,
            217,
            231,
            263,
            26,
            45,
            11,
            42,
            236,
            114,
            134,
            187,
            130,
            290,
            55,
            40,
            132,
            112,
            279,
            172,
            52,
            34,
            169,
            186,
            3,
            183,
            86,
            119,
            115,
            152,
            240,
            109,
            242,
            265,
            48,
            73,
            17,
            133,
            108,
            288,
            32,
            64,
            7,
            120,
            181,
            87,
            145,
            178,
            27,
            137,
            151,
            156,
            192,
            113,
            82,
            37,
            286,
            14,
            197,
            75,
            57,
        ]
    )
    train_lasso, test_lasso = (
        train_processed[:, lasso_feature_indices],
        test_processed[:, lasso_feature_indices],
    )  # Create Dataset 5: Lasso features

    (
        train_subset,
        test_subset,
        subset_cols,
    ) = (
        data.subsetted_data()
    )  # Access Dataset 3: same as dataset 2 but with overlapping columns dropped

    (
        train_curated,
        test_curated,
        curated_cols,
    ) = (
        data.curated_data()
    )  # Access Dataset 4: curated set; see list in class method defintion

    # Saving all files in pickled format for fast loading
    print("Saving raw data with shape {}".format(train_raw.shape))
    np.save(os.path.join(DATA_PATH, "train_raw.npy"), train_raw)
    np.save(os.path.join(DATA_PATH, "test_raw.npy"), test_raw)

    print("Saving full data with shape {}".format(train_processed.shape))
    np.save(os.path.join(DATA_PATH, "train_data_full.npy"), train_processed)
    np.save(os.path.join(DATA_PATH, "test_data_full.npy"), test_processed)

    print("Saving Lasso data with shape {}".format(train_lasso.shape))
    np.save(os.path.join(DATA_PATH, "train_lasso.npy"), train_lasso)
    np.save(os.path.join(DATA_PATH, "test_lasso.npy"), test_lasso)

    print("Saving subset data with shape {}".format(train_subset.shape))
    np.save(os.path.join(DATA_PATH, "train_data_subset.npy"), train_subset)
    np.save(os.path.join(DATA_PATH, "test_data_subset.npy"), test_subset)

    print("Saving curated data with shape {}".format(train_curated.shape))
    np.save(os.path.join(DATA_PATH, "train_data_curated.npy"), train_curated)
    np.save(os.path.join(DATA_PATH, "test_data_curated.npy"), test_curated)

    np.save(os.path.join(DATA_PATH, "train_labels.npy"), train_labels)

    return


if __name__ == "__main__":
    main()
