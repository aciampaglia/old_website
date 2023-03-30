async function run() {
    // Load and plot the original input data that we are going to train on.
    loadData()
  }
  document.addEventListener('DOMContentLoaded', run);
  function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.
  
    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      const inputs = data.map(d => d.horsepower)
      const labels = data.map(d => d.mpg);
  
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

const csvUrl = "/C:/Users/Administrator/Desktop/website/aciampaglia.github.io/database.csv";
async function loadData() {
   // We want to predict the column "medv", which represents a median value of
   // a home (in $1000s), so we mark it as a label.
   const csvDataset = tf.data.csv(
     csvUrl, {
       columnConfigs: {
         m1: {
           isLabel: true
         }, 
         m2: {
            isLabel: true
         },
         q1: {
            isLabel: true
         },
         q2: {
            isLabel: true
         },
       }
     });

   // Number of features is the number of column names minus one for the label
   // column.
   const numOfFeatures = (await csvDataset.columnNames()).length - 4;

   // Prepare the Dataset for training.
   const flattenedDataset =
     csvDataset
     .map(({xs, ys}) =>
       {
         // Convert xs(features) and ys(labels) from object form (keyed by
         // column name) to array form.
         return {xs:Object.values(xs), ys:Object.values(ys)};
       })
     .batch(40);

   // Define the model.
   // Define input, which has a size of 5 (not including batch dimension).
    const input = tf.input({shape: [numOfFeatures]});
    const denseLayer1 = tf.layers.dense({units: 100, activation: 'tanh'});
    const denseLayer2 = tf.layers.dense({units: 75, activation: 'tanh'});
    const denseLayer3 = tf.layers.dense({units: 50, activation: 'tanh'});
    // Obtain the output symbolic tensor by applying the layers on the input.
    const output = denseLayer3.apply(denseLayer2.apply(denseLayer1.apply(input)));
    // Create the model based on the inputs.
    const model = tf.model({inputs: input, outputs: output});

    model.compile({
        optimizer: tf.train.sgd(0.000001),
        loss: 'meanSquaredError'
    });

   // Fit the model using the prepared Dataset
   return model.fitDataset(flattenedDataset, {
     epochs: 10000,
     callbacks: {
       onEpochEnd: async (epoch, logs) => {
         console.log(epoch + ':' + logs.loss);
       }
     }
   });
}

  