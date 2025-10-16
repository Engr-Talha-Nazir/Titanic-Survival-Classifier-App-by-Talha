// Global variables
let trainData = null;
let testData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let testPredictions = null;
let rocData = null;

// DOM elements
const elements = {
    loadBtn: document.getElementById('load-btn'),
    preprocessBtn: document.getElementById('preprocess-btn'),
    buildModelBtn: document.getElementById('build-model-btn'),
    trainBtn: document.getElementById('train-btn'),
    predictBtn: document.getElementById('predict-btn'),
    downloadBtn: document.getElementById('download-btn'),
    saveModelBtn: document.getElementById('save-model-btn'),
    thresholdSlider: document.getElementById('threshold-slider'),
    thresholdValue: document.getElementById('threshold-value')
};

// Robust CSV parser that handles quoted fields and commas
function parseCSV(csvText) {
    const rows = [];
    let currentRow = [];
    let currentField = '';
    let inQuotes = false;
    
    for (let i = 0; i < csvText.length; i++) {
        const char = csvText[i];
        const nextChar = csvText[i + 1] || '';
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                // Escaped quote inside quotes
                currentField += '"';
                i++; // Skip next quote
            } else {
                // Toggle quote state
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // End of field
            currentRow.push(currentField);
            currentField = '';
        } else if ((char === '\n' || char === '\r') && !inQuotes) {
            // Handle line endings
            if (char === '\r' && nextChar === '\n') i++; // Skip \n after \r
            // End of row
            if (currentField !== '' || currentRow.length > 0) {
                currentRow.push(currentField);
                rows.push(currentRow);
            }
            currentRow = [];
            currentField = '';
        } else {
            currentField += char;
        }
    }
    
    // Add final row if exists
    if (currentField !== '' || currentRow.length > 0) {
        currentRow.push(currentField);
        rows.push(currentRow);
    }
    
    return rows;
}

// File loading and inspection
elements.loadBtn.addEventListener('click', async () => {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile) {
        alert('Please select a training CSV file');
        return;
    }
    
    try {
        const trainText = await trainFile.text();
        const trainRows = parseCSV(trainText);
        
        if (trainRows.length < 2) {
            throw new Error('Training file is empty or invalid');
        }
        
        const headers = trainRows[0];
        const dataRows = trainRows.slice(1);
        
        // Convert to array of objects
        trainData = dataRows.map(row => {
            const obj = {};
            headers.forEach((header, index) => {
                obj[header.trim()] = row[index] ? row[index].trim() : '';
            });
            return obj;
        });
        
        // Load test data if provided
        if (testFile) {
            const testText = await testFile.text();
            const testRows = parseCSV(testText);
            
            if (testRows.length > 1) {
                const testHeaders = testRows[0];
                const testDataRows = testRows.slice(1);
                
                testData = testDataRows.map(row => {
                    const obj = {};
                    testHeaders.forEach((header, index) => {
                        obj[header.trim()] = row[index] ? row[index].trim() : '';
                    });
                    return obj;
                });
            }
        }
        
        displayDataPreview(trainData, testData);
        calculateDataStats(trainData);
        
    } catch (error) {
        alert('Error loading files: ' + error.message);
        console.error(error);
    }
});

function displayDataPreview(trainData, testData) {
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '';
    
    // Training data preview
    const trainPreview = document.createElement('div');
    trainPreview.innerHTML = `<h3>Training Data Preview (${trainData.length} rows)</h3>`;
    
    if (trainData.length > 0) {
        const sample = trainData.slice(0, 5);
        const table = createTable(['#', ...Object.keys(sample[0])], 
            sample.map((row, i) => [i + 1, ...Object.values(row)]));
        trainPreview.appendChild(table);
    }
    
    // Test data preview
    if (testData) {
        const testPreview = document.createElement('div');
        testPreview.innerHTML = `<h3>Test Data Preview (${testData.length} rows)</h3>`;
        
        if (testData.length > 0) {
            const sample = testData.slice(0, 5);
            const table = createTable(['#', ...Object.keys(sample[0])], 
                sample.map((row, i) => [i + 1, ...Object.values(row)]));
            testPreview.appendChild(table);
        }
        previewDiv.appendChild(testPreview);
    }
    
    previewDiv.appendChild(trainPreview);
}

function calculateDataStats(data) {
    // Calculate missing values and basic stats
    const stats = {};
    const features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived'];
    
    features.forEach(feature => {
        const values = data.map(row => row[feature]).filter(val => val !== '');
        const missing = data.length - values.length;
        
        stats[feature] = {
            missing: missing,
            missingPercent: ((missing / data.length) * 100).toFixed(1),
            unique: [...new Set(values)].length
        };
    });
    
    // Display stats
    const statsTable = createTable(
        ['Feature', 'Missing', 'Missing %', 'Unique Values'],
        Object.entries(stats).map(([feature, stat]) => 
            [feature, stat.missing, stat.missingPercent + '%', stat.unique])
    );
    
    document.getElementById('data-preview').appendChild(statsTable);
    
    // Create survival charts
    createSurvivalCharts(data);
}

function createSurvivalCharts(data) {
    const survivalBySex = {};
    const survivalByPclass = {};
    
    data.forEach(passenger => {
        const survived = parseInt(passenger.Survived);
        const sex = passenger.Sex;
        const pclass = passenger.Pclass;
        
        if (!isNaN(survived)) {
            // Sex analysis
            if (!survivalBySex[sex]) survivalBySex[sex] = { survived: 0, total: 0 };
            survivalBySex[sex].total++;
            if (survived === 1) survivalBySex[sex].survived++;
            
            // Pclass analysis
            if (!survivalByPclass[pclass]) survivalByPclass[pclass] = { survived: 0, total: 0 };
            survivalByPclass[pclass].total++;
            if (survived === 1) survivalByPclass[pclass].survived++;
        }
    });
    
    // Create charts using tfjs-vis
    const sexData = {
        values: Object.entries(survivalBySex).map(([sex, stats]) => ({
            sex: sex,
            survival_rate: (stats.survived / stats.total) * 100
        }))
    };
    
    const pclassData = {
        values: Object.entries(survivalByPclass).map(([pclass, stats]) => ({
            pclass: `Class ${pclass}`,
            survival_rate: (stats.survived / stats.total) * 100
        }))
    };
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Data Analysis' },
        sexData,
        { xLabel: 'Sex', yLabel: 'Survival Rate %' }
    );
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Data Analysis' },
        pclassData,
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate %' }
    );
}

// Preprocessing
elements.preprocessBtn.addEventListener('click', () => {
    if (!trainData) {
        alert('Please load data first');
        return;
    }
    
    try {
        preprocessData();
    } catch (error) {
        alert('Error in preprocessing: ' + error.message);
        console.error(error);
    }
});

function preprocessData() {
    const outputDiv = document.getElementById('preprocess-output');
    outputDiv.innerHTML = '<h3>Preprocessing Steps:</h3>';
    
    // Schema definition - CHANGE THESE if your dataset has different columns
    const featureColumns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
    const targetColumn = 'Survived';
    const idColumn = 'PassengerId';
    
    let steps = [];
    
    // Step 1: Handle missing values
    steps.push('1. Handling missing values:');
    
    // Age - impute with median
    const ages = trainData.map(p => parseFloat(p.Age)).filter(age => !isNaN(age));
    const ageMedian = ages.length > 0 ? ages.reduce((a, b) => a + b) / ages.length : 30;
    steps.push(`   - Age: imputed ${trainData.filter(p => !p.Age || p.Age === '').length} missing values with median (${ageMedian.toFixed(1)})`);
    
    // Embarked - impute with mode
    const embarkedCounts = {};
    trainData.forEach(p => {
        if (p.Embarked && p.Embarked !== '') {
            embarkedCounts[p.Embarked] = (embarkedCounts[p.Embarked] || 0) + 1;
        }
    });
    const embarkedMode = Object.keys(embarkedCounts).reduce((a, b) => 
        embarkedCounts[a] > embarkedCounts[b] ? a : b, 'S');
    steps.push(`   - Embarked: imputed ${trainData.filter(p => !p.Embarked || p.Embarked === '').length} missing values with mode (${embarkedMode})`);
    
    // Fare - impute with median
    const fares = trainData.map(p => parseFloat(p.Fare)).filter(fare => !isNaN(fare));
    const fareMedian = fares.length > 0 ? fares.reduce((a, b) => a + b) / fares.length : 32;
    steps.push(`   - Fare: imputed ${trainData.filter(p => !p.Fare || p.Fare === '').length} missing values with median (${fareMedian.toFixed(2)})`);
    
    // Step 2: Create new features (optional) - COMMENT OUT if you don't want these features
    steps.push('2. Creating new features:');
    steps.push('   - FamilySize = SibSp + Parch + 1');
    steps.push('   - IsAlone = 1 if FamilySize == 1, else 0');
    
    // Step 3: Prepare features and labels
    steps.push('3. Preparing features and labels:');
    
    const processedData = trainData.map(passenger => {
        const processed = {};
        
        // Handle missing values
        processed.Age = passenger.Age && passenger.Age !== '' ? 
            parseFloat(passenger.Age) : ageMedian;
        processed.Embarked = passenger.Embarked && passenger.Embarked !== '' ? 
            passenger.Embarked : embarkedMode;
        processed.Fare = passenger.Fare && passenger.Fare !== '' ? 
            parseFloat(passenger.Fare) : fareMedian;
        
        // Copy other features
        processed.Pclass = parseInt(passenger.Pclass);
        processed.Sex = passenger.Sex;
        processed.SibSp = parseInt(passenger.SibSp);
        processed.Parch = parseInt(passenger.Parch);
        
        // New features - COMMENT OUT THE NEXT 2 LINES if you don't want these features
        processed.FamilySize = processed.SibSp + processed.Parch + 1;
        processed.IsAlone = processed.FamilySize === 1 ? 1 : 0;
        
        // Target
        processed.Survived = passenger.Survived !== '' ? 
            parseInt(passenger.Survived) : null;
        
        return processed;
    }).filter(p => p.Survived !== null); // Remove rows with missing target
    
    steps.push(`   - Final training samples: ${processedData.length}`);
    
    // Step 4: Calculate standardization parameters
    // Update this array based on which features you include
    const numericFeatures = ['Age', 'Fare']; // Remove 'FamilySize' if not using it
    const standardizationParams = {};
    
    numericFeatures.forEach(feature => {
        const values = processedData.map(p => p[feature]);
        standardizationParams[feature] = {
            mean: values.reduce((a, b) => a + b) / values.length,
            std: Math.sqrt(values.map(x => Math.pow(x - values.reduce((a, b) => a + b) / values.length, 2))
                .reduce((a, b) => a + b) / values.length)
        };
    });
    
    steps.push('4. Standardization parameters calculated');
    
    // Step 5: One-hot encoding mapping
    const categoricalFeatures = {
        Sex: ['male', 'female'],
        Pclass: ['1', '2', '3'],
        Embarked: ['C', 'Q', 'S']
    };
    
    steps.push('5. One-hot encoding prepared for: Sex, Pclass, Embarked');
    
    // Store preprocessing parameters for later use
    window.preprocessingParams = {
        ageMedian,
        embarkedMode,
        fareMedian,
        standardizationParams,
        categoricalFeatures,
        useFamilyFeatures: true // Set to false if not using FamilySize and IsAlone
    };
    
    window.processedTrainData = processedData;
    
    // Display steps
    steps.forEach(step => {
        const p = document.createElement('p');
        p.style.margin = '5px 0';
        p.textContent = step;
        outputDiv.appendChild(p);
    });
    
    outputDiv.innerHTML += `<p><strong>Preprocessing completed! Ready for model building.</strong></p>`;
}

// Model Building
elements.buildModelBtn.addEventListener('click', () => {
    if (!window.processedTrainData) {
        alert('Please preprocess data first');
        return;
    }
    
    try {
        buildModel();
    } catch (error) {
        alert('Error building model: ' + error.message);
        console.error(error);
    }
});

function buildModel() {
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Architecture:</h3>';
    
    // Calculate input shape based on features used
    // Base features: Pclass(3) + Sex(2) + Age(1) + SibSp(1) + Parch(1) + Fare(1) + Embarked(3) = 12
    let inputShape = 12;
    
    // Add FamilySize and IsAlone if used (2 more features)
    if (window.preprocessingParams.useFamilyFeatures) {
        inputShape += 2; // Total becomes 14
    }
    
    // Create sequential model
    model = tf.sequential();
    
    // Add layers with correct input shape
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape]
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const modelSummary = [];
    model.summary(1, 1, line => modelSummary.push(line));
    
    const pre = document.createElement('pre');
    pre.style.background = '#f5f5f5';
    pre.style.padding = '10px';
    pre.textContent = modelSummary.join('\n');
    summaryDiv.appendChild(pre);
    
    summaryDiv.innerHTML += `<p><strong>Input shape: [${inputShape}] - Adjust if you change feature selection</strong></p>`;
}

// Training
elements.trainBtn.addEventListener('click', async () => {
    if (!model || !window.processedTrainData) {
        alert('Please build model and preprocess data first');
        return;
    }
    
    try {
        await trainModel();
    } catch (error) {
        alert('Error training model: ' + error.message);
        console.error(error);
    }
});

async function trainModel() {
    const progressDiv = document.getElementById('training-progress');
    progressDiv.innerHTML = '<h3>Training in progress...</h3>';
    
    // Prepare features and labels
    const { features, labels } = prepareFeaturesAndLabels(window.processedTrainData);
    
    // Create 80/20 stratified split
    const splitIndex = Math.floor(features.shape[0] * 0.8);
    
    const trainFeatures = features.slice(0, splitIndex);
    const trainLabels = labels.slice(0, splitIndex);
    const valFeatures = features.slice(splitIndex);
    const valLabels = labels.slice(splitIndex);
    
    validationData = [valFeatures, valLabels];
    
    // Training callbacks
    const callbacks = {
        onEpochEnd: (epoch, logs) => {
            progressDiv.innerHTML = `
                <h3>Training Progress</h3>
                <p>Epoch ${epoch + 1}/50</p>
                <p>Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}</p>
                <p>Val Loss: ${logs.val_loss.toFixed(4)}, Val Accuracy: ${logs.val_acc.toFixed(4)}</p>
            `;
            
            // Update tfjs-vis charts
            tfvis.show.history({ name: 'Training History', tab: 'Training' }, 
                [{ epoch: epoch + 1, loss: logs.loss, val_loss: logs.val_loss, 
                   acc: logs.acc, val_acc: logs.val_acc }], 
                ['loss', 'val_loss', 'acc', 'val_acc']);
        }
    };
    
    // Train model
    trainingHistory = await model.fit(trainFeatures, trainLabels, {
        epochs: 50,
        batchSize: 32,
        validationData: validationData,
        callbacks: callbacks,
        verbose: 0
    });
    
    progressDiv.innerHTML += '<p><strong>Training completed!</strong></p>';
    
    // Calculate and display metrics
    calculateMetrics();
}

function prepareFeaturesAndLabels(data) {
    const params = window.preprocessingParams;
    const featuresArray = [];
    const labelsArray = [];
    
    data.forEach(passenger => {
        const featureVector = [];
        
        // One-hot encode Pclass
        const pclassOneHot = [0, 0, 0];
        pclassOneHot[parseInt(passenger.Pclass) - 1] = 1;
        featureVector.push(...pclassOneHot);
        
        // One-hot encode Sex
        const sexOneHot = passenger.Sex === 'female' ? [1, 0] : [0, 1];
        featureVector.push(...sexOneHot);
        
        // Standardize Age
        const standardizedAge = (passenger.Age - params.standardizationParams.Age.mean) / 
                              params.standardizationParams.Age.std;
        featureVector.push(standardizedAge);
        
        // SibSp (keep as is)
        featureVector.push(passenger.SibSp);
        
        // Parch (keep as is)
        featureVector.push(passenger.Parch);
        
        // Standardize Fare
        const standardizedFare = (passenger.Fare - params.standardizationParams.Fare.mean) / 
                               params.standardizationParams.Fare.std;
        featureVector.push(standardizedFare);
        
        // One-hot encode Embarked
        const embarkedOneHot = [0, 0, 0];
        const embarkedIndex = ['C', 'Q', 'S'].indexOf(passenger.Embarked);
        if (embarkedIndex !== -1) embarkedOneHot[embarkedIndex] = 1;
        featureVector.push(...embarkedOneHot);
        
        // Add FamilySize and IsAlone if using them
        if (params.useFamilyFeatures) {
            // Standardize FamilySize
            if (params.standardizationParams.FamilySize) {
                const standardizedFamilySize = (passenger.FamilySize - params.standardizationParams.FamilySize.mean) / 
                                            params.standardizationParams.FamilySize.std;
                featureVector.push(standardizedFamilySize);
            } else {
                featureVector.push(passenger.FamilySize); // Use raw value if not standardized
            }
            
            // IsAlone (binary, no standardization needed)
            featureVector.push(passenger.IsAlone);
        }
        
        featuresArray.push(featureVector);
        labelsArray.push(passenger.Survived);
    });
    
    return {
        features: tf.tensor2d(featuresArray),
        labels: tf.tensor1d(labelsArray)
    };
}

// Metrics calculation and display
function calculateMetrics() {
    if (!model || !validationData) return;
    
    const [valFeatures, valLabels] = validationData;
    const predictions = model.predict(valFeatures);
    const probs = predictions.dataSync();
    const trueLabels = valLabels.dataSync();
    
    // Calculate ROC curve data
    rocData = calculateROCCurve(probs, trueLabels);
    
    // Display ROC curve
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: rocData.points },
        { xLabel: 'False Positive Rate', yLabel: 'True Positive Rate' }
    );
    
    // Display AUC
    const metricsDiv = document.getElementById('performance-metrics');
    if (metricsDiv) {
        metricsDiv.innerHTML = `
            <h3>Model Performance</h3>
            <p><strong>AUC: ${rocData.auc.toFixed(4)}</strong></p>
        `;
    }
    
    // Initial metrics with default threshold
    updateMetrics(0.5);
}

function calculateROCCurve(probabilities, trueLabels) {
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const points = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < probabilities.length; i++) {
            const prediction = probabilities[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 1 && actual === 0) fp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 0 && actual === 1) fn++;
        }
        
        const tpr = tp + fn === 0 ? 0 : tp / (tp + fn);
        const fpr = fp + tn === 0 ? 0 : fp / (fp + tn);
        
        points.push({ x: fpr, y: tpr, threshold });
    });
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < points.length; i++) {
        auc += (points[i].x - points[i-1].x) * (points[i].y + points[i-1].y) / 2;
    }
    
    return { points, auc };
}

// Threshold slider event
elements.thresholdSlider.addEventListener('input', (e) => {
    const threshold = parseFloat(e.target.value);
    elements.thresholdValue.textContent = threshold.toFixed(2);
    updateMetrics(threshold);
});

function updateMetrics(threshold) {
    if (!rocData || !validationData) return;
    
    const [valFeatures, valLabels] = validationData;
    const predictions = model.predict(valFeatures);
    const probs = predictions.dataSync();
    const trueLabels = valLabels.dataSync();
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < probs.length; i++) {
        const prediction = probs[i] >= threshold ? 1 : 0;
        const actual = trueLabels[i];
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    
    // Update confusion matrix
    const confusionMatrixDiv = document.getElementById('confusion-matrix');
    if (confusionMatrixDiv) {
        confusionMatrixDiv.innerHTML = `
            <h3>Confusion Matrix (Threshold: ${threshold.toFixed(2)})</h3>
            ${createTable(
                ['', 'Predicted 0', 'Predicted 1'],
                [
                    ['Actual 0', tn, fp],
                    ['Actual 1', fn, tp]
                ]
            ).outerHTML}
        `;
    }
    
    // Update performance metrics
    const performanceDiv = document.getElementById('performance-metrics');
    if (performanceDiv) {
        // Preserve AUC display
        const aucDisplay = performanceDiv.querySelector('p') ? performanceDiv.querySelector('p').outerHTML : '';
        performanceDiv.innerHTML = aucDisplay + `
            <h3>Performance Metrics</h3>
            ${createTable(
                ['Metric', 'Value'],
                [
                    ['Accuracy', accuracy.toFixed(4)],
                    ['Precision', precision.toFixed(4)],
                    ['Recall', recall.toFixed(4)],
                    ['F1-Score', f1.toFixed(4)]
                ]
            ).outerHTML}
        `;
    }
}

// Prediction and Export
elements.predictBtn.addEventListener('click', async () => {
    if (!model || !testData) {
        alert('Please train model and load test data first');
        return;
    }
    
    try {
        await predictTestData();
    } catch (error) {
        alert('Error predicting: ' + error.message);
        console.error(error);
    }
});

async function predictTestData() {
    const resultsDiv = document.getElementById('prediction-results');
    resultsDiv.innerHTML = '<h3>Generating predictions...</h3>';
    
    // Preprocess test data using training parameters
    const processedTestData = testData.map(passenger => {
        const processed = {};
        const params = window.preprocessingParams;
        
        // Handle missing values using training parameters
        processed.Age = passenger.Age && passenger.Age !== '' ? 
            parseFloat(passenger.Age) : params.ageMedian;
        processed.Embarked = passenger.Embarked && passenger.Embarked !== '' ? 
            passenger.Embarked : params.embarkedMode;
        processed.Fare = passenger.Fare && passenger.Fare !== '' ? 
            parseFloat(passenger.Fare) : params.fareMedian;
        
        // Copy other features
        processed.Pclass = parseInt(passenger.Pclass);
        processed.Sex = passenger.Sex;
        processed.SibSp = parseInt(passenger.SibSp);
        processed.Parch = parseInt(passenger.Parch);
        processed.PassengerId = passenger.PassengerId;
        
        // New features - only if using them
        if (params.useFamilyFeatures) {
            processed.FamilySize = processed.SibSp + processed.Parch + 1;
            processed.IsAlone = processed.FamilySize === 1 ? 1 : 0;
        }
        
        return processed;
    });
    
    // Prepare features for prediction
    const testFeaturesArray = processedTestData.map(passenger => {
        const featureVector = [];
        const params = window.preprocessingParams;
        
        // One-hot encode Pclass
        const pclassOneHot = [0, 0, 0];
        pclassOneHot[parseInt(passenger.Pclass) - 1] = 1;
        featureVector.push(...pclassOneHot);
        
        // One-hot encode Sex
        const sexOneHot = passenger.Sex === 'female' ? [1, 0] : [0, 1];
        featureVector.push(...sexOneHot);
        
        // Standardize Age
        const standardizedAge = (passenger.Age - params.standardizationParams.Age.mean) / 
                              params.standardizationParams.Age.std;
        featureVector.push(standardizedAge);
        
        // SibSp
        featureVector.push(passenger.SibSp);
        
        // Parch
        featureVector.push(passenger.Parch);
        
        // Standardize Fare
        const standardizedFare = (passenger.Fare - params.standardizationParams.Fare.mean) / 
                               params.standardizationParams.Fare.std;
        featureVector.push(standardizedFare);
        
        // One-hot encode Embarked
        const embarkedOneHot = [0, 0, 0];
        const embarkedIndex = ['C', 'Q', 'S'].indexOf(passenger.Embarked);
        if (embarkedIndex !== -1) embarkedOneHot[embarkedIndex] = 1;
        featureVector.push(...embarkedOneHot);
        
        // Add FamilySize and IsAlone if using them
        if (params.useFamilyFeatures) {
            // FamilySize (standardize if parameters exist, otherwise use raw)
            if (params.standardizationParams.FamilySize) {
                const standardizedFamilySize = (passenger.FamilySize - params.standardizationParams.FamilySize.mean) / 
                                            params.standardizationParams.FamilySize.std;
                featureVector.push(standardizedFamilySize);
            } else {
                featureVector.push(passenger.FamilySize);
            }
            
            // IsAlone (binary)
            featureVector.push(passenger.IsAlone);
        }
        
        return featureVector;
    });
    
    const testFeatures = tf.tensor2d(testFeaturesArray);
    const predictions = model.predict(testFeatures);
    testPredictions = predictions.dataSync();
    
    // Display sample predictions
    const samplePredictions = processedTestData.slice(0, 10).map((passenger, i) => [
        passenger.PassengerId,
        testPredictions[i].toFixed(4),
        testPredictions[i] >= 0.5 ? '1' : '0'
    ]);
    
    resultsDiv.innerHTML = `
        <h3>Test Predictions (First 10)</h3>
        ${createTable(
            ['PassengerId', 'Probability', 'Prediction'],
            samplePredictions
        ).outerHTML}
        <p>Total predictions generated: ${testPredictions.length}</p>
    `;
}

elements.downloadBtn.addEventListener('click', () => {
    if (!testPredictions || !testData) {
        alert('Please generate predictions first');
        return;
    }
    
    // Create submission CSV
    let csvContent = 'PassengerId,Survived\n';
    const threshold = parseFloat(elements.thresholdSlider.value);
    
    testData.forEach((passenger, i) => {
        const prediction = testPredictions[i] >= threshold ? 1 : 0;
        csvContent += `${passenger.PassengerId},${prediction}\n`;
    });
    
    // Create probabilities CSV
    let probContent = 'PassengerId,Probability\n';
    testData.forEach((passenger, i) => {
        probContent += `${passenger.PassengerId},${testPredictions[i].toFixed(6)}\n`;
    });
    
    // Download files
    downloadCSV(csvContent, 'submission.csv');
    downloadCSV(probContent, 'probabilities.csv');
});

elements.saveModelBtn.addEventListener('click', async () => {
    if (!model) {
        alert('Please train model first');
        return;
    }
    
    try {
        await model.save('downloads://titanic-model');
    } catch (error) {
        alert('Error saving model: ' + error.message);
        console.error(error);
    }
});

// Utility functions
function createTable(headers, rows) {
    const table = document.createElement('table');
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    rows.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    return table;
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', filename);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}
