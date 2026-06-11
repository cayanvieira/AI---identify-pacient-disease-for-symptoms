import * as tf from '@tensorflow/tfjs';

let _model = null;
let _globalCtx = {}

const normalize = (value, min, max) => (value - min) / (max - min) || 1

const WEIGHTS = {
    symptoms: 0.5,
    smoker:0.4,
    age: 0.2,
    weight: 0.1,
}

const oneHotWeighted = (index, length, weight) => {
    return tf.oneHot(index, length).cast("float32").mul(weight)
}
const multiHotWeighted = (listIndex, length, weight) => {
    const multiHot = tf.zeros([length])
   
    const result = listIndex.reduce((acc, index) => {
        return acc.add(oneHotWeighted(index, length, weight))
    }, multiHot)
    return result
}

function makeContext(pacients, diseases) {
    const ages = pacients.map(pacient => pacient.age);
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const weigth = pacients.map(pacient => pacient.weight);
    const minWeight = Math.min(...weigth);
    const maxWeight = Math.max(...weigth);


    const symptoms = [...new Set(diseases.map(disease => disease.symptoms).flat())]

    const symptomsIndex = Object.fromEntries(
        symptoms.map((symptom, index) => [symptom, index])
    );

    const midAge = (minAge + maxAge) / 2;
    const midWeight = (minWeight + maxWeight) / 2;


    const pacientAgesSums = {}
    const pacientCounts = {}

    const pacientWeightSums = {}
    const pacientWeightCounts = {}

    diseases.forEach(disease => {
        pacients.forEach(pacient => {
            if (pacient.diseaseNameValue === disease.value) {
                pacientAgesSums[disease.value] = (pacientAgesSums[disease.value] || 0) + pacient.age;
                pacientCounts[disease.value] = (pacientCounts[disease.value] || 0) + 1;
                pacientWeightSums[disease.value] = (pacientWeightSums[disease.value] || 0) + pacient.weight;
                pacientWeightCounts[disease.value] = (pacientWeightCounts[disease.value] || 0) + 1;
            }
        })
    })



    const diseasesAverages = Object.fromEntries(
        diseases.map(disease => {
            const avgAge = pacientAgesSums[disease.value]
                ? pacientAgesSums[disease.value] / pacientCounts[disease.value]
                : midAge
            const avgWeight = pacientWeightSums[disease.value] ?
                pacientWeightSums[disease.value] / pacientWeightCounts[disease.value]
                : midWeight
            return [disease.value, { normalizeAge: normalize(avgAge, minAge, maxAge), normalizeWeight: normalize(avgWeight, minWeight, maxWeight) }]
        })
    )

    return {
        pacients,
        diseases,
        minAge,
        maxAge,
        minWeight,
        maxWeight,
        midAge,
        midWeight,
        symptomsIndex,
        diseasesAverages,
        symptomsCount: symptoms.length,
        //age+weight+smoker+symptoms
        dimensions: 3 + symptoms.length
    }

}
//função que transforma um produto em um vetor de características numéricas, usando as informações do contexto para normalizar e codificar as características do produto
function  encoderDisease(disease, context) {
   
    const age = tf.tensor1d([(context.diseasesAverages[disease.value].normalizeAge ?? 0.5) * WEIGHTS.age])
    const weight = tf.tensor1d([(context.diseasesAverages[disease.value].normalizeWeight ?? 0.5) * WEIGHTS.weight])
    const smoker =  disease.highlyAssociatedWithSmoking ? tf.tensor1d([WEIGHTS.smoker]) : tf.zeros([1])
    const indexs = context.diseases.filter((item)=> item.value === disease.value)[0].symptoms.map(symptom => context.symptomsIndex[symptom])
    const symptoms = multiHotWeighted(indexs, context.symptomsCount, WEIGHTS.symptoms)

    return tf.concat([age, weight, smoker, symptoms])
}

function encoderPacient(pacient, context) {
    if (pacient.symptoms.length) {
        const age = tf.tensor1d([normalize(pacient.age, context.minAge, context.maxAge) * WEIGHTS.age]);
        const weight = tf.tensor1d([normalize(pacient.weight, context.minWeight, context.maxWeight) * WEIGHTS.weight]);
        const smoker = tf.tensor1d([(pacient.smoker ? 1 : 0) * WEIGHTS.smoker]); // Exemplo de peso para fumante
        
        const indexs = pacient.symptoms.map(symptom => context.symptomsIndex[symptom])
        const symptoms = multiHotWeighted(indexs, context.symptomsCount, WEIGHTS.symptoms);
        
        return tf.concat([age, weight, smoker, symptoms]);
    }

    return tf.concat1d([
        tf.tensor1d([normalize(pacient.age, context.minAge, context.maxAge) * WEIGHTS.age]),
        tf.zeros([1]),//weight is not informed for pacient, so we will use 0 and the model will learn to ignore this feature for pacient vectors, but consider it for diseases vectors
        tf.zeros([1]),//smoker is not informed for pacient, so we will use 0 and the model will learn to ignore this feature for pacient vectors, but consider it for diseases vectors   
        tf.zeros([context.symptomsCount]),
    ]).reshape([1, context.dimensions])

}
function createTrainingData(context) {
    const inputs = [];
    const labels = [];

    context.pacients.filter(pacient => pacient.symptoms.length).forEach(pacient => {
        const pacientVector = Array.from(encoderPacient(pacient, context).dataSync());
        context.diseases.forEach(disease => {
            const diseaseVector = Array.from(encoderDisease(disease, context).dataSync());
            const label = pacient.diseaseNameValue === disease.value ? 1 : 0;
            inputs.push([...pacientVector, ...diseaseVector])
            labels.push(label);
        })
    })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor1d(labels),
        inputDimention: context.dimensions * 2,
    }
}
async function configureNeuralNetAndTrain(trainingData) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [trainingData.inputDimention], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    await model.fit(trainingData.xs, trainingData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            // onEpochEnd: (epoch, logs) => {
            //     console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.accuracy.toFixed(4)}`);
            // }
        }
    })
    return model;
}

async function trainModel(pacients, diseases) {
    const context = makeContext(pacients, diseases)

    context.diseaseVectors = diseases.map(disease => {
        return {
            name: disease.name,
            meta: { ...disease },
            vector: encoderDisease(disease, context).dataSync()
        }
    })

    _globalCtx = context
    
    const trainingData = await createTrainingData(context);
    
    _model = await configureNeuralNetAndTrain(trainingData);

    return self.postMessage({ action: 'TRAIN_COMPLETE', success: true });
}

function recommend(pacient) {
    if (!_model) {
        return
    }

    const pacientVector = encoderPacient(pacient, _globalCtx).dataSync()
    const inputs = _globalCtx.diseaseVectors.map(disease => [...pacientVector, ...disease.vector])

    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor)
    const scores = predictions.dataSync();
    const recommendations = _globalCtx.diseaseVectors
        .map((disease, index) => ({ ...disease.meta, name: disease.value, score: scores[index] }))
        .sort((a, b) => b.score - a.score).filter((item,i)=> i<3)


    const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score);
    postMessage({ action: 'ANALISED_PACIENT', diseaseMatch: sortedRecommendations, pacient });

}

const handlers = {
    ['START_TRAINING']: data => trainModel(data.pacients,data.diseases),
    ['ANALISING_PACIENT']: data => {recommend(data.pacient)},
}

self.onmessage = async function (e) {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};


