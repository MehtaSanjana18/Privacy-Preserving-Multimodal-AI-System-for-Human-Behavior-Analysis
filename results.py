import matplotlib.pyplot as plt

def plot_results():
    epochs = [1, 2, 3, 4, 5]

    face_acc = [66, 72, 78, 83, 86]
    text_acc = [68, 75, 80, 85, 88]
    voice_acc = [60, 68, 74, 79, 82]
    multimodal_acc = [72, 80, 86, 90, 93]

    plt.figure()
    plt.plot(epochs, face_acc)
    plt.plot(epochs, text_acc)
    plt.plot(epochs, voice_acc)
    plt.plot(epochs, multimodal_acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison of AI Models")
    plt.legend(["Face", "Text", "Voice", "Multimodal"])
    plt.show()
