package main;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ann.Shape;
import ann.Tensor;

//includes utility functions
public class Utility {
	// reads the main set of MNIST images (provided the filepaths for the images and
	// labels file)
	// reads based off the encoding described in the MNIST doc
	// (https://yann.lecun.com/exdb/mnist)
	// returns in tensor form
	public static Tensor[] readSet(String imagesFile, String labelsFile) {
		try {
		    // Open label file
		    FileInputStream fin = new FileInputStream(new File(labelsFile));
		    ByteBuffer buff = ByteBuffer.wrap(fin.readAllBytes());
		    buff.order(ByteOrder.BIG_ENDIAN); // Ensure correct byte order
	
		    // Read magic number (not needed)
		    buff.getInt();
		    // Read dataset size
		    int NUM_IMAGES = buff.getInt();
		    
		    // Allocate label tensor (NUM_IMAGES, 10) for one-hot encoding
		    Tensor labelsTensor = new Tensor(new Shape(NUM_IMAGES, 10));
		    labelsTensor.init(); // Initialize with zeros
		    
		    // Read labels and one-hot encode
		    for (int i = 0; i < NUM_IMAGES; i++) {
		        int label = (int) buff.get();
		        labelsTensor.set(new int[]{label, i}, 1.0f);
		    }
		    fin.close();
	
		    // Open image file
		    fin = new FileInputStream(new File(imagesFile));
		    buff = ByteBuffer.wrap(fin.readAllBytes());
		    buff.order(ByteOrder.BIG_ENDIAN);
	
		    // Read magic number (not needed)
		    buff.getInt();
		    // Read dataset size
		    NUM_IMAGES = buff.getInt();
		    // Read image dimensions (28x28)
		    int sizeX = buff.getInt();
		    int sizeY = buff.getInt();
		    int imageSize = sizeX * sizeY; // 784 pixels per image
	
		    // Allocate image tensor (NUM_IMAGES, 784)
		    Tensor imagesTensor = new Tensor(new Shape(NUM_IMAGES, imageSize));
		    imagesTensor.init();
	
		    // Read and normalize images
		    for (int i = 0; i < NUM_IMAGES; i++) {
		        for (int j = 0; j < imageSize; j++) {
		            byte pixel = buff.get();
		            float normalizedPixel = (pixel & 0xFF) / 255.0f; // Normalize to [0, 1]
		            imagesTensor.set(new int[]{j, i}, normalizedPixel);
		        }
		    }
		    fin.close();
	
		    // Return tensors
		    return new Tensor[]{imagesTensor, labelsTensor};
		} catch(Exception e) {
			System.err.println("Something went wrong reading the files");
			e.printStackTrace();
		}
		return null;
	}

}
