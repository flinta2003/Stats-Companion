import unittest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from project import descriptive_stat, prediction, data_reader, cleaning


class TestProject(unittest.TestCase):    
    def setUp(self):
        self.train_data = pd.read_excel("continuous_train.xlsx")
        self.test_data = pd.read_excel("becsles.xlsx")
        self.iris_categorical = pd.read_csv("Iris.csv")
        self.iris_becs = pd.read_excel("iris_becs.xlsx")
    

    def test_data_reader(self):
        train_output1, test_output1 = data_reader("continuous_train.xlsx","becsles.xlsx")
        self.assertTrue(train_output1.equals(self.train_data))
        self.assertTrue(test_output1.equals(self.test_data))

        train_output2, test_output2 = data_reader("iris.csv", "iris_becs.xlsx")
        self.assertTrue(train_output2.equals(self.iris_categorical))
        self.assertTrue(test_output2.equals(self.iris_becs))

        train_output3, test_output3 = data_reader("continuous_train.xlsx",0)
        self.assertTrue(train_output3.equals(self.train_data))
        self.assertEqual(test_output3, 0)

        train_output3, test_output3 = data_reader("iris.csv", 0)
        self.assertTrue(train_output3.equals(self.iris_categorical))
        self.assertEqual(test_output3, 0)


    def test_cleaning(self):
        #binary
        input_binary = pd.read_excel("wine_binary.xlsx")
        _, func_X1, func_y1, __ = cleaning(tanito = input_binary, y_variable = "quality", make_dummies = True, y_type = "binary")
        output_binary1 = pd.read_excel("test_wine_binary.xlsx")
        output_y1 = output_binary1["quality_Premium"]
        output_X1 = output_binary1.drop(columns= "quality_Premium")
        assert_frame_equal(func_X1, output_X1, check_dtype=False, check_like=True)
        assert_series_equal(func_y1, output_y1, check_dtype=False, check_like=True)

        #continuous
        func_train, _, __, func_test = cleaning(tanito = self.train_data, X_test = self.test_data, y_variable = "exam_score", make_dummies = True)
        output_test = pd.read_excel("test_becsles.xlsx")
        output_train = pd.read_excel("test_continuous_train.xlsx")
        assert_frame_equal(func_train, output_train, check_dtype = False, check_like = True)
        assert_frame_equal(func_test, output_test, check_dtype = False, check_like = True)

        #categorical
        _, func_X2, func_y2, func_test2 = cleaning(tanito = self.iris_categorical, X_test = self.iris_becs, y_variable = "Species", make_dummies = True, y_type= "categorical")
        output_train2 = pd.read_excel("test_Iris.xlsx")
        output_X2 = output_train2.drop(columns = "Species")
        output_y2 = output_train2["Species"]
        output_test2 = pd.read_excel("test_iris_becs.xlsx")
        assert_frame_equal(func_test2, output_test2, check_dtype = False, check_like = True)
        assert_frame_equal(func_X2, output_X2, check_dtype = False, check_like = True)
        assert_series_equal(func_y2, output_y2, check_dtype = False, check_like = True)

        #descriptive stat
        func_train2, _, __, ___ = cleaning(tanito = self.train_data)
        output_train3 = pd.read_excel("test_descriptive_stat.xlsx")
        assert_frame_equal(func_train2, output_train3, check_dtype = False, check_like = True)


    def test_descriptive_stat(self):
        #continuous
        func_output = descriptive_stat(self.train_data, self.train_data["exam_score"], "exam_score")
        test_output = "\nDescriptive Statistical Analysis:\nThe distribution of exam_score is slightly right-skewed and flatter than a normal. The highest value is 51.300 and the lowest is 17.100,\nin addition half of values is higher and half is lower than the 34.100 value." \
        "The average value is 33.964, \nwhile the average difference from the mean value is 6.788. The data middle 50% is between 29.500 and 38.900."
        self.assertEqual(func_output, test_output)
        test_indicators = pd.DataFrame({"Indicator": ["Mean", "Standard deviation", "Median", "Minimum", "Maximum", "Range", "Relative Standard Deviation", "Interquartile Range", "Skewness", "Kurtosis"],
                                       "Value": [33.964, 6.788, 34.1, 17.1, 51.3, 34.2, 5.003, 9.4,  0.029, -0.358]})
        func_indicators = pd.read_csv("stat_results_exam_score")
        assert_frame_equal(test_indicators, func_indicators, check_like = True, check_dtype = False)

        #categorical
        desc_stat_cat = pd.read_excel("categorical_train.xlsx")
        func_output2 = descriptive_stat(desc_stat_cat, desc_stat_cat["Grade"], "Grade")
        test_output2 = "\nStatistical Analysis:\nB is the most common value in variable Grade with 81 instances. Your vairable has 5 categories altogether."
        self.assertEqual(func_output2, test_output2)

if __name__ == "__main__":
    unittest.main()