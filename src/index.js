// Import TensorFlow.js library
const tf = require('@tensorflow/tfjs-node');

class ModelDeployment {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.model = null;
  }

  async loadModel() {
    try {
      this.model = await tf.loadLayersModel(`file://${this.modelPath}`);
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
    }
  }

  async predict(inputData) {
    if (!this.model) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    try {
      const inputTensor = tf.tensor(inputData);
      const predictions = await this.model.predict(inputTensor).data();
      return predictions;
    } catch (error) {
      console.error('Error making predictions:', error);
      throw error;
    }
  }
}

module.exports = ModelDeployment;
