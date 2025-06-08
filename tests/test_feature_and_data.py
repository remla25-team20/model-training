from lib_ml import preprocessing


class TestFeaturesAndData:
    def test_preprocessing_function(self):
        assert preprocessing._text_process("green is not red") == "green not red"
