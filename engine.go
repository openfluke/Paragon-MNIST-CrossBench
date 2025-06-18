package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/openfluke/paragon/v3"
	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

const (
	epochs       = 10
	learningRate = 0.05
	batchSize    = 64
	modelsDir    = "./models"
)

func main() {
	startTotal := time.Now()
	fmt.Println("🚀 Running experiment: MNIST")

	// Create models directory
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		fmt.Printf("❌ Failed to create models directory: %v\n", err)
		return
	}

	// Load MNIST data
	fmt.Println("⚙ Stage: MNIST Dataset Prep")
	startData := time.Now()
	mnist := experiments.NewMNISTDatasetStage("./data/mnist")
	exp := pilot.NewExperiment("MNIST", mnist)
	if err := exp.RunAll(); err != nil {
		fmt.Println("❌ Experiment failed:", err)
		os.Exit(1)
	}
	allInputs, allTargets, err := loadMNISTData("./data/mnist")
	if err != nil {
		fmt.Println("❌ Failed to load MNIST:", err)
		return
	}
	fmt.Printf("📊 Dataset sizes: Train=%d, Test=%d\n", len(allInputs)*8/10, len(allInputs)*2/10)
	fmt.Printf("⏱ Data Prep Time: %v\n", time.Since(startData))

	// Split into 80% training and 20% testing
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(allInputs, allTargets, 0.8)

	// Build the network
	startInit := time.Now()
	nn := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{{28, 28}, {32, 32}, {10, 1}},
		[]string{"linear", "relu", "softmax"},
		[]bool{true, true, true},
	)
	fmt.Printf("⏱ Network Init Time: %v\n", time.Since(startInit))

	// Enable WebGPU
	nn.WebGPUNative = true
	nn.Debug = false // Disable debugging
	startGPU := time.Now()
	if err := nn.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️ Failed to initialize WebGPU: %v\n", err)
		fmt.Println("   Continuing with CPU-only processing...")
		nn.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
		defer nn.CleanupOptimizedGPU()
	}
	fmt.Printf("⏱ WebGPU Init Time: %v\n", time.Since(startGPU))

	// Train the network
	fmt.Println("🧠 Training the network...")
	startTrain := time.Now()
	nn.TrainWithGPUSync(trainInputs, trainTargets, epochs, learningRate, false, float32(2), float32(-2))
	fmt.Printf("⏱ Total Training Time: %v\n", time.Since(startTrain))

	// Evaluate with ADHD
	startEval := time.Now()
	trainScore := evaluateFullNetwork(nn, trainInputs, trainTargets, "Train")
	testScore := evaluateFullNetwork(nn, testInputs, testTargets, "Test")
	fmt.Printf("📊 Train Score: %.4f%%\n", trainScore)
	fmt.Printf("📊 Test Score: %.4f%%\n", testScore)
	fmt.Printf("⏱ Evaluation Time: %v\n", time.Since(startEval))

	// Save the model
	startSave := time.Now()
	modelPath := filepath.Join(modelsDir, "mnist_model.json")
	if err := nn.SaveJSON(modelPath); err != nil {
		fmt.Printf("❌ Failed to save model: %v\n", err)
	} else {
		fmt.Printf("💾 Saved model to %s\n", modelPath)
	}
	fmt.Printf("⏱ Model Save Time: %v\n", time.Since(startSave))
	fmt.Printf("⏱ Total Experiment Time: %v\n", time.Since(startTotal))
}

func evaluateFullNetwork[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64, dataset string) float64 {
	start := time.Now()
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))

	for i := range inputs {
		nn.Forward(inputs[i])
		out := nn.ExtractOutput()
		expected[i] = float64(paragon.ArgMax(targets[i][0]))
		actual[i] = float64(paragon.ArgMax(out))
	}

	nn.EvaluateModel(expected, actual)
	score := nn.Performance.Score

	// Print ADHD assessment
	fmt.Printf("\n📈 ADHD Performance (%s Set):\n", dataset)
	for name, bucket := range nn.Performance.Buckets {
		fmt.Printf("- %s: %d samples (%.2f%%)\n", name, bucket.Count, float64(bucket.Count)/float64(nn.Performance.Total)*100)
	}
	fmt.Printf("- Total Samples: %d\n", nn.Performance.Total)
	fmt.Printf("- Failures (100%%+): %d (%.2f%%)\n", nn.Performance.Failures, float64(nn.Performance.Failures)/float64(nn.Performance.Total)*100)
	fmt.Printf("- Score: %.4f%%\n", score)
	fmt.Printf("⏱ Evaluate Time (%s): %v\n", dataset, time.Since(start))

	return score
}
