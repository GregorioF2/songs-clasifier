from pyAudioAnalysis import audioTrainTest as aT
aT.extract_features_and_train(["./samples/like_themes/", "./samples/unlike_themes/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "liked", True)

print('\n\n Clasiffie archivo')

aT.file_classification("./samples/test_themes/test_1_like.wav", "./liked", "knn")
