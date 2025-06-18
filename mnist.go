package main

import (
	"encoding/binary"
	"os"

	"path/filepath"
)

// Loads both training and test images, returns as one dataset
func loadMNISTData(dir string) ([][][]float64, [][][]float64, error) {
	images := make([][][]float64, 0)
	labels := make([][][]float64, 0)

	for _, set := range []string{"train", "t10k"} {
		imgPath := filepath.Join(dir, set+"-images-idx3-ubyte")
		lblPath := filepath.Join(dir, set+"-labels-idx1-ubyte")

		imgs, err := loadMNISTImages(imgPath)
		if err != nil {
			return nil, nil, err
		}

		lbls, err := loadMNISTLabels(lblPath)
		if err != nil {
			return nil, nil, err
		}

		images = append(images, imgs...)
		labels = append(labels, lbls...)
	}

	return images, labels, nil
}

func loadMNISTImages(path string) ([][][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [16]byte
	if _, err := f.Read(header[:]); err != nil {
		return nil, err
	}
	num := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	images := make([][][]float64, num)
	buf := make([]byte, rows*cols)
	for i := 0; i < num; i++ {
		if _, err := f.Read(buf); err != nil {
			return nil, err
		}
		img := make([][]float64, rows)
		for r := 0; r < rows; r++ {
			img[r] = make([]float64, cols)
			for c := 0; c < cols; c++ {
				img[r][c] = float64(buf[r*cols+c]) / 255.0
			}
		}
		images[i] = img
	}
	return images, nil
}

func loadMNISTLabels(path string) ([][][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [8]byte
	if _, err := f.Read(header[:]); err != nil {
		return nil, err
	}
	num := int(binary.BigEndian.Uint32(header[4:8]))

	labels := make([][][]float64, num)
	for i := 0; i < num; i++ {
		var b [1]byte
		if _, err := f.Read(b[:]); err != nil {
			return nil, err
		}
		labels[i] = labelToOneHot(int(b[0]))
	}
	return labels, nil
}

func labelToOneHot(label int) [][]float64 {
	t := make([][]float64, 1)
	t[0] = make([]float64, 10)
	t[0][label] = 1.0
	return t
}
