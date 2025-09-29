/* Titanic Binary Classifier — TensorFlow.js (Browser)
 * Target: Survived (0/1)
 * Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
 * Exclude: PassengerId
 * Notes: robust CSV parser (handles quotes/commas), tfjs-vis charts, ROC/AUC.
 */

// ---------- DOM ----------
const els = {
  trainFile: document.getElementById('trainFile'),
  testFile: document.getElementById('testFile'),
  btnLoad: document.getElementById('btnLoad'),
  previewMeta: document.getElementById('previewMeta'),
  previewTable: document.getElementById('previewTable'),
  loadInfo: document.getElementById('loadInfo'),

  toggleFamily: document.getElementById('toggleFamily'),
  btnPreprocess: document.getElementById('btnPreprocess'),
  prepInfo: document.getElementById('prepInfo'),

  btnBuild: document.getElementById('btnBuild'),
  visModel: document.getElementById('visModel'),
  modelSummary: document.getElementById('modelSummary'),
  btnTrain: document.getElementById('btnTrain'),
  btnEvaluate: document.getElementById('btnEvaluate'),
  visFit: document.getElementById('visFit'),
  trainInfo: document.getElementById('trainInfo'),

  thrSlider: document.getElementById('thrSlider'),
  thrVal: document.getElementById('thrVal'),
  btnUpdateThr: document.getElementById('btnUpdateThr'),
  cmStats: document.getElementById('cmStats'),
  prfStats: document.getElementById('prfStats'),
  visROC: document.getElementById('visROC'),
  aucInfo: document.getElementById('aucInfo'),

  btnPredict: document.getElementById('btnPredict'),
  btnSaveModel: document.getElementById('btnSaveModel'),
  predInfo: document.getElementById('predInfo'),
  btnExportSubmission: document.getElementById('btnExportSubmission'),
  btnExportProbs: document.getElementById('btnExportProbs'),

  visSex: document.getElementById('visSex'),
  visPclass: document.getElementById('visPclass'),
};

// ---------- Globals ----------
let rawTrain=null, rawTest=null;
let X=null, y=null, featureNames=[];
let medians={}, modes={}, scalers={};
let useFamily=true;
let model=null, split={}, valProbs=null, valTrue=null, roc=null, threshold=0.5;
let testProbs=null, submission=null, probabilities=null;

// ---------- CSV (fixed: quotes/commas) ----------
function parseCSV(text){
  const rows=[]; let i=0, field='', inQuotes=false, row=[];
  const pushField=()=>{ row.push(field); field=''; };
  const pushRow =()=>{ if(row.length){ rows.push(row); row=[]; }};

  while(i<text.length){
    const c=text[i];
    if(inQuotes){
      if(c === '"'){ if(text[i+1] === '"'){ field+='"'; i++; } else { inQuotes=false; } }
      else field+=c;
    } else {
      if(c === '"') inQuotes=true;
      else if(c === ',') pushField();
      else if(c === '\n' || c === '\r'){ if(c==='\r' && text[i+1]==='\n') i++; pushField(); pushRow(); }
      else field+=c;
    }
    i++;
  }
  if(field.length || row.length){ pushField(); pushRow(); }
  while(rows.length && rows[rows.length-1].every(v=>v==='')) rows.pop();

  const header = rows.shift().map(h=>h.trim());
  return rows.map(r=>{
    const o={};
    header.forEach((h,idx)=>{
      const v = r[idx]===undefined || r[idx]==='' ? null : r[idx];
      o[h] = (v!==null && !isNaN(v)) ? Number(v) : v; // numeric if possible
    });
    return o;
  });
}
function readFile(inputEl){
  return new Promise((resolve,reject)=>{
    const file=inputEl.files?.[0];
    if(!file) return resolve(null);
    const reader=new FileReader();
    reader.onload=()=>resolve(reader.result);
    reader.onerror=e=>reject(e);
    reader.readAsText(file);
  });
}

// ---------- Helpers ----------
function renderPreview(tableEl, rows, limit=12){
  if(!rows?.length){ tableEl.innerHTML='<tr><td class="muted">No rows.</td></tr>'; return; }
  const header=Object.keys(rows[0]);
  const thead=`<thead><tr>${header.map(h=>`<th>${h}</th>`).join('')}</tr></thead>`;
  const tbody=`<tbody>${rows.slice(0,limit).map(r=>`<tr>${header.map(h=>`<td>${String(r[h]??'').slice(0,80)}</td>`).join('')}</tr>`).join('')}</tbody>`;
  tableEl.innerHTML=thead+tbody;
}
function missingPercent(rows, cols){
  const total=rows.length; const out={};
  for(const c of cols){ let m=0; for(const r of rows){ if(r[c]==null || r[c]==='') m++; } out[c]= total? (100*m/total):0; }
  return out;
}
function median(arr){ const a=arr.filter(v=>v!=null && !Number.isNaN(v)).slice().sort((x,y)=>x-y); if(!a.length) return 0; const m=Math.floor(a.length/2); return a.length%2?a[m]:(a[m-1]+a[m])/2; }
function mode(arr){ const f=new Map(); let best=null,c=-1; for(const v of arr){ if(v==null || v==='') continue; f.set(v,(f.get(v)||0)+1); if(f.get(v)>c){c=f.get(v); best=v;} } return best; }
function meanStd(arr){ const a=arr.filter(v=>v!=null && !Number.isNaN(v)); if(!a.length) return {mean:0,std:1}; const m=a.reduce((s,v)=>s+v,0)/a.length; const vv=a.reduce((s,v)=>s+(v-m)*(v-m),0)/Math.max(1,a.length-1); return {mean:m,std:Math.max(1e-8,Math.sqrt(vv))}; }
function oneHot(val,cats){ const v=new Array(cats.length).fill(0); const idx=cats.indexOf(val); if(idx>=0) v[idx]=1; return v; }
function confusion(yTrue,yProb,thr){ let TP=0,TN=0,FP=0,FN=0; for(let i=0;i<yTrue.length;i++){ const p=yProb[i]>=thr?1:0; const t=yTrue[i]; if(p===1&&t===1) TP++; else if(p===1&&t===0) FP++; else if(p===0&&t===0) TN++; else FN++; } return {TP,TN,FP,FN}; }
function prf({TP,FP,FN}){ const precision=TP+FP===0?0:TP/(TP+FP); const recall=TP+FN===0?0:TP/(TP+FN); const f1=(precision+recall)===0?0:2*(precision*recall)/(precision+recall); return {precision,recall,f1}; }
function computeROC(yTrue,yProb){ const steps=200; const pts=[]; for(let i=0;i<=steps;i++){ const thr=i/steps; let TP=0,TN=0,FP=0,FN=0; for(let j=0;j<yTrue.length;j++){ const p=yProb[j]>=thr?1:0; const t=yTrue[j]; if(p===1&&t===1) TP++; else if(p===1&&t===0) FP++; else if(p===0&&t===0) TN++; else FN++; } const TPR=TP+FN?TP/(TP+FN):0; const FPR=FP+TN?FP/(FP+TN):0; pts.push([FPR,TPR]); }
  pts.sort((a,b)=>a[0]-b[0]); let auc=0; for(let i=1;i<pts.length;i++){ const [x0,y0]=pts[i-1]; const [x1,y1]=pts[i]; auc+=(x1-x0)*(y0+y1)/2; } return {fpr:pts.map(p=>p[0]), tpr:pts.map(p=>p[1]), auc}; }
function renderKeyVals(el,obj){ el.innerHTML=Object.entries(obj).map(([k,v])=>`<div>${k}: <strong>${typeof v==='number'?v.toFixed(4):v}</strong></div>`).join(''); }

// ---------- Schema ----------
const TARGET='Survived';
const ID_COL='PassengerId';
const FEATURE_COLS=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'];
const ONEHOT_CATS={ Sex:['male','female'], Pclass:['1','2','3'], Embarked:['C','Q','S'] };
const NUMERIC_Z=['Age','Fare'];

// ---------- Load & Inspect ----------
async function handleLoad(){
  try{
    const trainText=await readFile(els.trainFile);
    const testText =await readFile(els.testFile);
    if(!trainText){ alert('Please choose train.csv'); return; }

    rawTrain=parseCSV(trainText);
    rawTest = testText? parseCSV(testText) : null;

    renderPreview(els.previewTable, rawTrain);
    const miss=missingPercent(rawTrain,[TARGET,ID_COL,...FEATURE_COLS]);
    els.previewMeta.innerHTML=`Train rows: <strong>${rawTrain.length}</strong> · Test rows: <strong>${rawTest?rawTest.length:0}</strong><br><span class="muted">Missing% → ${Object.entries(miss).map(([k,v])=>`${k}:${v.toFixed(1)}%`).join(' · ')}</span>`;
    els.loadInfo.textContent='Loaded. Next: Preprocessing → Build → Train.';

    renderSurvivalBars(rawTrain);
  }catch(e){ console.error(e); alert('Failed to load CSV.'); }
}
function renderSurvivalBars(rows){
  const bySex={}, byClass={};
  for(const r of rows){
    const s=r.Sex, p=r.Pclass, y=Number(r.Survived);
    if(s!=null){ bySex[s]??={surv:0,total:0}; bySex[s].total++; if(y===1) bySex[s].surv++; }
    if(p!=null){ byClass[p]??={surv:0,total:0}; byClass[p].total++; if(y===1) byClass[p].surv++; }
  }
  const sexData=Object.entries(bySex).map(([k,v])=>({name:k,value:v.total? v.surv/v.total : 0}));
  const pclassData=Object.entries(byClass).map(([k,v])=>({name:`P${k}`,value:v.total? v.surv/v.total : 0}));
  tfvis.render.barchart(els.visSex, sexData, {yLabel:'Survival Rate', xLabel:'Sex', width:420, height:240});
  tfvis.render.barchart(els.visPclass, pclassData, {yLabel:'Survival Rate', xLabel:'Pclass', width:420, height:240});
}

// ---------- Preprocessing ----------
function preprocessRows(rows, fitStats=null, isTrain=true, useFamilyLocal=true){
  if(isTrain){
    medians.Age=median(rows.map(r=>Number(r.Age)));
    medians.Fare=median(rows.map(r=>Number(r.Fare)));
    modes.Embarked=mode(rows.map(r=>r.Embarked));
  }
  const ageMed=medians.Age??28, fareMed=medians.Fare??14.45, embMode=modes.Embarked??'S';

  const clean=rows.map(r=>{
    const o={...r};
    if(o.Age==null) o.Age=ageMed;
    if(o.Fare==null) o.Fare=fareMed;
    if(!o.Embarked) o.Embarked=embMode;
    return o;
  });

  let zStats=fitStats;
  if(isTrain){
    zStats={};
    for(const c of NUMERIC_Z){ zStats[c]=meanStd(clean.map(r=>Number(r[c]))); }
  }
  for(const r of clean){
    for(const c of NUMERIC_Z){ const v=Number(r[c]); r[c]=(v-zStats[c].mean)/zStats[c].std; }
  }

  const featCols=[...FEATURE_COLS];
  if(useFamilyLocal){
    for(const r of clean){ const f=(r.SibSp||0)+(r.Parch||0)+1; r.FamilySize=f; r.IsAlone=(f===1)?1:0; }
    featCols.push('FamilySize','IsAlone');
  }

  featureNames=[];
  for(const c of featCols){
    if(ONEHOT_CATS[c]) for(const k of ONEHOT_CATS[c]) featureNames.push(`${c}_${k}`);
    else featureNames.push(c);
  }

  const Xarr=[], yarr=[];
  for(const r of clean){
    const vec=[];
    for(const c of featCols){
      if(ONEHOT_CATS[c]){
        const raw = r[c]==null ? '' : String(r[c]);
        vec.push(...oneHot(raw, ONEHOT_CATS[c]));
      } else vec.push(Number(r[c]));
    }
    Xarr.push(vec);
    if(r[TARGET]!=null) yarr.push(Number(r[TARGET]));
  }
  return { Xt: tf.tensor2d(Xarr), yt: yarr.length? tf.tensor1d(yarr,'int32'): null, zStats };
}
function stratifiedSplit(Xt, yt, testSize=0.2){
  const yData=Array.from(yt.dataSync()); const idx0=[], idx1=[];
  yData.forEach((v,i)=> (v===1?idx1:idx0).push(i));
  function take(idxs){ const n=idxs.length, nTest=Math.max(1,Math.round(n*testSize)); const sh=idxs.slice().sort(()=>Math.random()-0.5); return {test:sh.slice(0,nTest), train:sh.slice(nTest)}; }
  const s0=take(idx0), s1=take(idx1); const trainIdx=s0.train.concat(s1.train), testIdx=s0.test.concat(s1.test);
  const gather=(t,indices)=>tf.tidy(()=>tf.gather(t, tf.tensor1d(indices,'int32')));
  return { X_train:gather(Xt,trainIdx), y_train:gather(yt,trainIdx), X_val:gather(Xt,testIdx), y_val:gather(yt,testIdx) };
}

// ---------- Model ----------
function buildModel(inputDim){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:16,activation:'relu',inputShape:[inputDim]}));
  m.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  m.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  return m;
}

// ---------- Events ----------
els.btnLoad.addEventListener('click', handleLoad);

els.btnPreprocess.addEventListener('click', ()=>{
  if(!rawTrain){ alert('Load train.csv first.'); return; }
  useFamily = !!els.toggleFamily.checked;
  const {Xt, yt, zStats} = preprocessRows(rawTrain, null, true, useFamily);
  X?.dispose(); y?.dispose(); X=Xt; y=yt; scalers=zStats;
  els.prepInfo.innerHTML=`Features: <strong>${featureNames.join(', ')}</strong><br/>X: <strong>${X.shape.join('×')}</strong>, y: <strong>${y.shape[0]}</strong>`;
});

els.btnBuild.addEventListener('click', ()=>{
  if(!X||!y){ alert('Run Preprocessing first.'); return; }
  model?.dispose(); model=buildModel(X.shape[1]);
  tfvis.show.modelSummary(els.visModel, model);
  const lines=[]; model.summary(80,undefined,l=>lines.push(l));
  els.modelSummary.innerHTML=`<pre style="white-space:pre-wrap;">${lines.join('\n')}</pre>`;
});

els.btnTrain.addEventListener('click', async ()=>{
  if(!model){ alert('Build model first.'); return; }
  if(split.X_train){ split.X_train.dispose(); split.y_train.dispose(); split.X_val.dispose(); split.y_val.dispose(); }
  split = stratifiedSplit(X, y, 0.2);

  const visCb = tfvis.show.fitCallbacks(els.visFit, ['loss','val_loss','acc','val_acc'], {callbacks:['onEpochEnd']});
  const statusCb = { onEpochEnd:(_,logs)=>{ els.trainInfo.textContent = `loss ${logs.loss.toFixed(4)} | acc ${logs.acc.toFixed(4)} | val_loss ${logs.val_loss.toFixed(4)} | val_acc ${logs.val_acc.toFixed(4)}`; } };
  const earlyStop = tf.callbacks.earlyStopping({monitor:'val_loss', patience:5, restoreBestWeight:true});

  await model.fit(split.X_train, split.y_train, {
    epochs:50, batchSize:32, validationData:[split.X_val, split.y_val],
    callbacks:[visCb, statusCb, earlyStop]    // single callbacks array (no overwrite)
  });
  els.trainInfo.textContent += ' — done.';
});

els.btnEvaluate.addEventListener('click', async ()=>{
  if(!split.X_val){ alert('Train first.'); return; }
  const probsT = model.predict(split.X_val);
  valProbs = Array.from(await probsT.data());  // flattened 1D
  probsT.dispose();
  valTrue = Array.from(await split.y_val.data());

  roc = computeROC(valTrue, valProbs);
  els.aucInfo.innerHTML = `AUC: <strong>${roc.auc.toFixed(4)}</strong>`;
  const rocPts = roc.fpr.map((x,i)=>({x, y: roc.tpr[i]}));
  tfvis.render.linechart(els.visROC, {values:rocPts, series:['ROC']}, {xLabel:'FPR', yLabel:'TPR', width:420, height:260});

  threshold = Number(els.thrSlider.value); els.thrVal.textContent = threshold.toFixed(2);
  updateThresholdStats();
});

function updateThresholdStats(){
  if(!valProbs || !valTrue) return;
  const cm = confusion(valTrue, valProbs, threshold);
  const m = prf(cm);
  renderKeyVals(els.cmStats, cm);
  renderKeyVals(els.prfStats, {Precision:m.precision, Recall:m.recall, F1:m.f1});
}
els.btnUpdateThr.addEventListener('click', ()=>{ threshold=Number(els.thrSlider.value); els.thrVal.textContent=threshold.toFixed(2); updateThresholdStats(); });
els.thrSlider.addEventListener('input', ()=>{ els.thrVal.textContent=Number(els.thrSlider.value).toFixed(2); });

els.btnPredict.addEventListener('click', async ()=>{
  if(!rawTest){ alert('Upload test.csv first.'); return; }
  if(!model){ alert('Build & train model first.'); return; }
  const {Xt} = preprocessRows(rawTest, scalers, false, useFamily);
  const probsT=model.predict(Xt);
  const probs=Array.from(await probsT.data());
  probsT.dispose(); Xt.dispose();
  testProbs=probs;

  const ids=rawTest.map(r=>Number(r[ID_COL]));
  submission = ids.map((id,i)=>({PassengerId:id, Survived: probs[i]>=threshold?1:0}));
  probabilities = ids.map((id,i)=>({PassengerId:id, Probability: probs[i]}));
  els.predInfo.textContent=`Predicted ${ids.length} rows @ threshold ${threshold.toFixed(2)}.`;
});

els.btnSaveModel.addEventListener('click', async ()=>{
  if(!model){ alert('No model.'); return; }
  await model.save('downloads://titanic-tfjs');
});

els.btnExportSubmission.addEventListener('click', ()=>{
  if(!submission){ alert('Run Predict first.'); return; }
  downloadCSV('submission.csv', submission.map(r=>[r.PassengerId,r.Survived]), ['PassengerId','Survived']);
});
els.btnExportProbs.addEventListener('click', ()=>{
  if(!probabilities){ alert('Run Predict first.'); return; }
  downloadCSV('probabilities.csv', probabilities.map(r=>[r.PassengerId,r.Probability]), ['PassengerId','Probability']);
});

// ---------- CSV download helper ----------
function downloadCSV(filename, rows, header){
  const lines=[];
  if(header) lines.push(header.join(','));
  for(const r of rows){
    const escaped = r.map(v=>{ const s=String(v??''); return /[",\n]/.test(s)? `"${s.replace(/"/g,'""')}"` : s; });
    lines.push(escaped.join(','));
  }
  const blob=new Blob([lines.join('\n')],{type:'text/csv;charset=utf-8;'}); const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}

// ---------- Cleanup ----------
window.addEventListener('beforeunload', ()=>{
  X?.dispose(); y?.dispose();
  if(split.X_train) split.X_train.dispose();
  if(split.y_train) split.y_train.dispose();
  if(split.X_val) split.X_val.dispose();
  if(split.y_val) split.y_val.dispose();
});
