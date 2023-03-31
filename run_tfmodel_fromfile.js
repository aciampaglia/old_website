async function run() {
    // Load and plot the original input data that we are going to train on.
    await tfvis.visor().close()
    const csvDataset = await loadData();
    const numOfFeatures = (await csvDataset.columnNames()).length - numOfLables;
    const model = await buildModel(numOfFeatures);
    await trainModel(model, csvDataset)

  }

  document.addEventListener('DOMContentLoaded', run);
  
// Hyper-parameters
const csvUrl = "database2.csv";
const batchSize = 40
const nEpochs = 300
const numOfLables = 4;

// Functions
async function loadData() {
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
         }
       }
     });

     return csvDataset;
    }

async function buildModel(numOfFeatures){
   // Define the model.
   // Define input, which has a size of 5 (not including batch dimension).
    const input = tf.input({shape: [numOfFeatures,]});
    const denseLayer1 = tf.layers.dense({units: 100, activation: 'tanh'});
    const denseLayer2 = tf.layers.dense({units: 75, activation: 'tanh'});
    const denseLayer3 = tf.layers.dense({units: 50, activation: 'tanh'});
    const outputLayer = tf.layers.dense({units: numOfLables, activation: 'linear'});
    // Obtain the output symbolic tensor by applying the layers on the input.
    const output = outputLayer.apply(denseLayer3.apply(denseLayer2.apply(denseLayer1.apply(input))));
    // Create the model based on the inputs.
    const model = tf.model({inputs: input, outputs: output});
    const surface = {name: 'Model Summary', tab: 'Model Inspection'};
    tfvis.show.modelSummary(surface, model);

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: ['mse']
    });

    return await model;
}
    async function trainModel(model, csvDataset){
   // Prepare the Dataset for training.
    const flattenedDataset =
     csvDataset
     .map(({xs, ys}) =>
       {
         // Convert xs(features) and ys(labels) from object form (keyed by
         // column name) to array form.
         return {xs:Object.values(xs), ys:Object.values(ys)};
       })
     .batch(batchSize);

   // Fit the model using the prepared Dataset
   const container = {name: 'Training process', tab: 'Model Inspection'}
   const metrics = ['loss']
   const opt = {callbacks:['onEpochEnd']}
   const fitCallbacks = tfvis.show.fitCallbacks(container, metrics, opt);
   const consoleCallback = {onEpochEnd: async (epoch, logs) => {
    console.log(epoch + ':' + logs.loss);
   }}
   return await model.fitDataset(flattenedDataset, {
     epochs: nEpochs,
     callbacks: [fitCallbacks, consoleCallback] 
/*      {
       onEpochEnd: async (epoch, logs) => {
         console.log(epoch + ':' + logs.loss);
         tfvis.show.fitCallbacks(
            {name: 'Training process', tab: 'Model Inspection'},
            ['loss'],
            { height: 200, callbacks: ['onEpochEnd']}
          )
        }
     }
 */   });
}

async function predict(model, inputData) {
    const [xs, preds] = tf.tidy(() => {

        const xsNorm = tf.linspace(4, 8, 100);
        const predictions = model.predict(xsNorm.reshape([100, 1]));
    
        const unNormXs = xsNorm
          .mul(inputMax.sub(inputMin))
          .add(inputMin);
    
        const unNormPreds = predictions
          .mul(labelMax.sub(labelMin))
          .add(labelMin);
    
        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
      });

}