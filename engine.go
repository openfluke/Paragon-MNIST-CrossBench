package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"time"

	"github.com/openfluke/paragon"
	"github.com/openfluke/pilot/experiments"
)

const (
	modelPath    = "mnist_model_float32.json"
	dataPath     = "./data/mnist"
	epochs       = 3
	learningRate = 0.01
	trainLimit   = 10 // 🔧 How many samples to train on
)

var layers = []struct{ Width, Height int }{
	{28, 28}, {32, 32}, {32, 32}, {10, 1},
}
var activations = []string{"linear", "relu", "relu", "softmax"}
var fullyConnected = []bool{true, true, true, true}

func main() {
	fmt.Println("🚀 Preparing MNIST Dataset...")

	mnist := experiments.NewMNISTDatasetStage(dataPath)
	if err := mnist.Run(); err != nil {
		fmt.Println("❌ MNIST setup failed:", err)
		return
	}

	fmt.Println("📦 Loading MNIST data into memory...")
	inputs, targets, err := LoadMNISTFloat32(dataPath, true)
	if err != nil {
		fmt.Println("❌ Data loading error:", err)
		return
	}

	// Truncate to trainLimit
	if trainLimit > 0 && trainLimit < len(inputs) {
		inputs = inputs[:trainLimit]
		targets = targets[:trainLimit]
	}

	var net *paragon.Network[float32]

	if fileExists(modelPath) {
		fmt.Println("📂 Loading saved model from JSON...")
		net = new(paragon.Network[float32])
		if err := net.LoadJSON(modelPath); err != nil {
			fmt.Println("❌ Failed to load model JSON:", err)
			return
		}
	} else {
		fmt.Println("🧠 Training float32 model...")
		net = paragon.NewNetwork[float32](layers, activations, fullyConnected)
		net.Train(inputs, targets, epochs, learningRate, false, 1.0, -1.0)

		fmt.Println("💾 Saving trained model...")
		if err := net.SaveJSON(modelPath); err != nil {
			fmt.Println("❌ Failed to save model JSON:", err)
			return
		}
	}

	sample := inputs[0]

	fmt.Println("\n⏱️ Benchmarking inference...")
	net.WebGPUNative = false
	cpuStart := time.Now()
	for i := 0; i < 1000; i++ {
		net.Forward(sample)
	}
	cpuElapsed := time.Since(cpuStart)

	net.WebGPUNative = true
	net.InitializeOptimizedGPU()
	net.Forward(sample) // warm-up
	gpuStart := time.Now()
	for i := 0; i < 1000; i++ {
		net.Forward(sample)
	}
	gpuElapsed := time.Since(gpuStart)
	net.CleanupOptimizedGPU()

	fmt.Printf("\n✅ CPU: %v\n", cpuElapsed)
	fmt.Printf("✅ GPU: %v\n", gpuElapsed)
	fmt.Printf("⚡ Speedup: %.2fx\n", float64(cpuElapsed)/float64(gpuElapsed))
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func LoadMNISTFloat32(path string, training bool) ([][][]float64, [][][]float64, error) {
	var imageFile, labelFile string
	if training {
		imageFile = path + "/train-images-idx3-ubyte"
		labelFile = path + "/train-labels-idx1-ubyte"
	} else {
		imageFile = path + "/t10k-images-idx3-ubyte"
		labelFile = path + "/t10k-labels-idx1-ubyte"
	}

	images, err := readImages(imageFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed reading images: %w", err)
	}

	labelsRaw, err := readLabels(labelFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed reading labels: %w", err)
	}

	labels := make([][][]float64, len(labelsRaw))
	for i, label := range labelsRaw {
		labels[i] = make([][]float64, 1)
		labels[i][0] = make([]float64, 10)
		labels[i][0][label] = 1.0
	}

	return images, labels, nil
}

func readImages(filename string) ([][][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, num, rows, cols int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &num)
	binary.Read(file, binary.BigEndian, &rows)
	binary.Read(file, binary.BigEndian, &cols)

	images := make([][][]float64, num)
	for i := 0; i < int(num); i++ {
		image := make([][]float64, rows)
		for r := 0; r < int(rows); r++ {
			image[r] = make([]float64, cols)
			for c := 0; c < int(cols); c++ {
				var pixel uint8
				binary.Read(file, binary.BigEndian, &pixel)
				image[r][c] = float64(pixel) / 255.0
			}
		}
		images[i] = image
	}
	return images, nil
}

func readLabels(filename string) ([]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, num int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &num)

	labels := make([]int, num)
	for i := 0; i < int(num); i++ {
		var label uint8
		binary.Read(file, binary.BigEndian, &label)
		labels[i] = int(label)
	}
	return labels, nil
}
