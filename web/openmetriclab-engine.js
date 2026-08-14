(function(global){
'use strict';
const EPS=1e-12;
const OML={};

OML.parseCSV=function(text){
  const rows=[]; let row=[], field='', q=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(q){ if(c==='"' && text[i+1]==='"'){field+='"';i++;} else if(c==='"'){q=false;} else field+=c; }
    else if(c==='"') q=true;
    else if(c===','){row.push(field);field='';}
    else if(c==='\n'){row.push(field.replace(/\r$/,'')); rows.push(row); row=[]; field='';}
    else field+=c;
  }
  if(field.length||row.length){row.push(field.replace(/\r$/,''));rows.push(row);}
  while(rows.length && rows[rows.length-1].every(x=>x==='')) rows.pop();
  if(!rows.length) return {headers:[],rows:[]};
  const headers=rows[0].map((h,i)=>h.trim()||`column_${i+1}`);
  return {headers,rows:rows.slice(1).filter(r=>r.some(x=>x!=='' )).map(r=>Object.fromEntries(headers.map((h,i)=>[h,r[i]??''])))};
};

function finitePairs(y,p){ const a=[],b=[]; for(let i=0;i<Math.min(y.length,p.length);i++){const x=+y[i],z=+p[i];if(Number.isFinite(x)&&Number.isFinite(z)){a.push(x);b.push(z);}} return [a,b]; }
function mean(a){return a.length?a.reduce((s,x)=>s+x,0)/a.length:NaN;}
function quantile(a,q){if(!a.length)return NaN;const b=[...a].sort((x,y)=>x-y),pos=(b.length-1)*q,lo=Math.floor(pos),hi=Math.ceil(pos);return lo===hi?b[lo]:b[lo]+(b[hi]-b[lo])*(pos-lo);}
function variance(a,m=mean(a)){return a.length?mean(a.map(x=>(x-m)*(x-m))):NaN;}
function correlation(a,b){const ma=mean(a),mb=mean(b);let n=0,da=0,db=0;for(let i=0;i<a.length;i++){const x=a[i]-ma,y=b[i]-mb;n+=x*y;da+=x*x;db+=y*y;}return da>0&&db>0?n/Math.sqrt(da*db):null;}
function ranks(a){const idx=a.map((v,i)=>[v,i]).sort((x,y)=>x[0]-y[0]);const out=Array(a.length);let i=0;while(i<idx.length){let j=i+1;while(j<idx.length&&idx[j][0]===idx[i][0])j++;const r=(i+j-1)/2+1;for(let k=i;k<j;k++)out[idx[k][1]]=r;i=j;}return out;}
OML.regressionMetrics=function(y0,p0){
  const [y,p]=finitePairs(y0,p0); if(y.length<2)throw new Error('At least two finite truth/prediction pairs are required.');
  const e=p.map((v,i)=>v-y[i]),ae=e.map(Math.abs),se=e.map(v=>v*v),my=mean(y),mp=mean(p);
  const sst=y.reduce((s,v)=>s+(v-my)*(v-my),0),sse=se.reduce((a,b)=>a+b,0);
  const nz=y.map(v=>Math.abs(v)>EPS);const mapeVals=ae.filter((_,i)=>nz[i]).map((v,i2)=>{let idx=-1,c=-1;for(let j=0;j<nz.length;j++){if(nz[j])c++;if(c===i2){idx=j;break;}}return v/Math.abs(y[idx]);});
  const sm=ae.map((v,i)=>{const d=Math.abs(y[i])+Math.abs(p[i]);return d>EPS?2*v/d:0;});
  const wden=y.reduce((s,v)=>s+Math.abs(v),0);
  return {n:y.length,mae:mean(ae),mse:mean(se),rmse:Math.sqrt(mean(se)),medianAE:quantile(ae,.5),maxError:Math.max(...ae),r2:sst>EPS?1-sse/sst:null,explainedVariance:variance(y)>EPS?1-variance(e)/variance(y):null,bias:mean(e),smape:100*mean(sm),mape:mapeVals.length?100*mean(mapeVals):null,mapeCoverage:mapeVals.length/y.length,wape:wden>EPS?100*ae.reduce((a,b)=>a+b,0)/wden:null,pearson:correlation(y,p),spearman:correlation(ranks(y),ranks(p)),p50:quantile(ae,.5),p90:quantile(ae,.9),p95:quantile(ae,.95),p99:quantile(ae,.99),y,p,error:e,absError:ae};
};

function unique(a){return [...new Set(a.map(String))];}
function cmatrix(y,p,classes){const k=classes.length,idx=new Map(classes.map((c,i)=>[c,i])),m=Array.from({length:k},()=>Array(k).fill(0));for(let i=0;i<y.length;i++){const a=idx.get(String(y[i])),b=idx.get(String(p[i]));if(a!=null&&b!=null)m[a][b]++;}return m;}
function rocBinary(y,score,pos){const pairs=y.map((v,i)=>({y:String(v)===String(pos)?1:0,s:+score[i]})).filter(o=>Number.isFinite(o.s)).sort((a,b)=>b.s-a.s);const P=pairs.reduce((s,o)=>s+o.y,0),N=pairs.length-P;if(!P||!N)return null;let tp=0,fp=0,prev=Infinity;const pts=[[0,0,Infinity]];for(const o of pairs){if(o.s!==prev){pts.push([fp/N,tp/P,prev]);prev=o.s;}if(o.y)tp++;else fp++;}pts.push([fp/N,tp/P,prev],[1,1,-Infinity]);let auc=0;for(let i=1;i<pts.length;i++)auc+=(pts[i][0]-pts[i-1][0])*(pts[i][1]+pts[i-1][1])/2;return {points:pts.map(x=>({fpr:x[0],tpr:x[1],threshold:x[2]})),auc};}
function prBinary(y,score,pos){const pairs=y.map((v,i)=>({y:String(v)===String(pos)?1:0,s:+score[i]})).filter(o=>Number.isFinite(o.s)).sort((a,b)=>b.s-a.s);const P=pairs.reduce((s,o)=>s+o.y,0);if(!P)return null;let tp=0,fp=0,ap=0,lastRecall=0;const pts=[{recall:0,precision:1,threshold:Infinity}];for(const o of pairs){if(o.y)tp++;else fp++;const recall=tp/P,precision=tp/(tp+fp);if(o.y){ap+=(recall-lastRecall)*precision;lastRecall=recall;}pts.push({recall,precision,threshold:o.s});}return {points:pts,ap};}
function mccMulticlass(m){const k=m.length;let c=0,s=0;const t=Array(k).fill(0),p=Array(k).fill(0);for(let i=0;i<k;i++)for(let j=0;j<k;j++){const v=m[i][j];s+=v;t[i]+=v;p[j]+=v;if(i===j)c+=v;}let tp=0,t2=0,p2=0;for(let i=0;i<k;i++){tp+=t[i]*p[i];t2+=t[i]*t[i];p2+=p[i]*p[i];}const num=c*s-tp,den=Math.sqrt((s*s-p2)*(s*s-t2));return den>0?num/den:0;}
OML.classificationMetrics=function(y0,p0,probabilities=null,probClasses=null){
  const n=Math.min(y0.length,p0.length),y=y0.slice(0,n).map(String),p=p0.slice(0,n).map(String),classes=unique([...y,...p,...((probClasses||[]).map(String))]);if(n<2)throw new Error('At least two classification rows are required.');
  const m=cmatrix(y,p,classes),per=[];let correct=0;const recalls=[];for(let i=0;i<classes.length;i++){const tp=m[i][i],fn=m[i].reduce((a,b)=>a+b,0)-tp,fp=m.reduce((s,r)=>s+r[i],0)-tp,sup=tp+fn;correct+=tp;const prec=tp+fp?tp/(tp+fp):0,rec=sup?tp/sup:0,f1=prec+rec?2*prec*rec/(prec+rec):0;recalls.push(rec);per.push({class:classes[i],precision:prec,recall:rec,f1,support:sup});}
  const macroF1=mean(per.map(x=>x.f1)),weightedF1=per.reduce((s,x)=>s+x.f1*x.support,0)/n,balanced=mean(recalls),acc=correct/n;
  const rowTotals=m.map(r=>r.reduce((a,b)=>a+b,0)),colTotals=classes.map((_,j)=>m.reduce((s,r)=>s+r[j],0));const pe=rowTotals.reduce((s,v,i)=>s+v*colTotals[i],0)/(n*n),kappa=Math.abs(1-pe)>EPS?(acc-pe)/(1-pe):0;
  const out={n,classes,accuracy:acc,balancedAccuracy:balanced,f1Macro:macroF1,f1Weighted:weightedF1,mcc:mccMulticlass(m),kappa,confusion:m,perClass:per};
  if(probabilities && probabilities.length===n){
    const pc=(probClasses||classes).map(String);const mat=probabilities.map(r=>r.map(Number));const eps=1e-15;let ll=0,brier=0;for(let i=0;i<n;i++){const s=mat[i].reduce((a,b)=>a+(Number.isFinite(b)?Math.max(0,b):0),0)||1;const norm=mat[i].map(v=>Math.max(eps,Math.min(1-eps,(Number.isFinite(v)?Math.max(0,v):0)/s)));const yi=pc.indexOf(y[i]);if(yi>=0){ll-=Math.log(norm[yi]);for(let j=0;j<pc.length;j++)brier+=(norm[j]-(j===yi?1:0))**2;}}
    out.logLoss=ll/n;out.brier=brier/n;
    if(classes.length===2){const pos=classes[1],j=pc.indexOf(pos);if(j>=0){const scores=mat.map(r=>+r[j]);out.roc=rocBinary(y,scores,pos);out.pr=prBinary(y,scores,pos);out.rocAuc=out.roc?.auc??null;out.averagePrecision=out.pr?.ap??null;out.positiveClass=pos;out.scores=scores;}}
    else {const aucs=[],aps=[];for(const c of classes){const j=pc.indexOf(c);if(j<0)continue;const scores=mat.map(r=>+r[j]),roc=rocBinary(y,scores,c),pr=prBinary(y,scores,c);if(roc)aucs.push(roc.auc);if(pr)aps.push(pr.ap);}out.rocAucMacro=aucs.length?mean(aucs):null;out.averagePrecisionMacro=aps.length?mean(aps):null;}
  }
  return out;
};
OML.binaryAtThreshold=function(y,scores,pos,thr){const p=scores.map(v=>+v>=thr?String(pos):'__NEG__'),negLabel=unique(y.map(String)).find(c=>c!==String(pos))||'0';return OML.classificationMetrics(y,p.map(v=>v==='__NEG__'?negLabel:v));};
OML.calibrationBinary=function(y,scores,pos,bins=10){const groups=Array.from({length:bins},()=>({sumP:0,sumY:0,n:0}));for(let i=0;i<Math.min(y.length,scores.length);i++){const p=Math.max(0,Math.min(1,+scores[i]));if(!Number.isFinite(p))continue;const b=Math.min(bins-1,Math.floor(p*bins));groups[b].sumP+=p;groups[b].sumY+=String(y[i])===String(pos)?1:0;groups[b].n++;}return groups.map((g,i)=>({bin:i,n:g.n,predicted:g.n?g.sumP/g.n:null,observed:g.n?g.sumY/g.n:null})).filter(g=>g.n);};

OML.imageToLabels=function(imageData,mode='auto',threshold=127){const {data,width,height}=imageData,n=width*height;let gray=true;for(let i=0;i<n;i++){if(data[4*i]!==data[4*i+1]||data[4*i]!==data[4*i+2]){gray=false;break;}}const vals=new Uint32Array(n);const seen=new Set();for(let i=0;i<n;i++){let v;if(gray){v=data[4*i];if(mode==='binary'||(mode==='auto'&&threshold!=null))v=v>threshold?1:0;}else{v=(data[4*i]<<16)|(data[4*i+1]<<8)|data[4*i+2];}vals[i]=v;seen.add(v);}return {width,height,labels:vals,classes:[...seen].sort((a,b)=>a-b),gray};};
function binaryMaskStats(a,b,lab,width,height,spacingX=1,spacingY=1){let tp=0,fp=0,fn=0,tn=0;for(let i=0;i<a.length;i++){const A=a[i]===lab,B=b[i]===lab;if(A&&B)tp++;else if(!A&&B)fp++;else if(A&&!B)fn++;else tn++;}const div=(n,d)=>d?n/d:null;const surface=(arr)=>{const pts=[];for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=y*width+x;if(arr[i]!==lab)continue;let edge=x===0||y===0||x===width-1||y===height-1;if(!edge){edge=arr[i-1]!==lab||arr[i+1]!==lab||arr[i-width]!==lab||arr[i+width]!==lab;}if(edge)pts.push([x*spacingX,y*spacingY]);}return pts;};const A=surface(a),B=surface(b);let sampled=false;const cap=4000;function sample(pts){if(pts.length<=cap)return pts;sampled=true;const out=[];const step=pts.length/cap;for(let i=0;i<cap;i++)out.push(pts[Math.floor(i*step)]);return out;}const As=sample(A),Bs=sample(B);function nearest(P,Q){if(!P.length||!Q.length)return [];const out=[];for(const p of P){let best=Infinity;for(const q of Q){const dx=p[0]-q[0],dy=p[1]-q[1],d=dx*dx+dy*dy;if(d<best)best=d;}out.push(Math.sqrt(best));}return out;}const d=[...nearest(As,Bs),...nearest(Bs,As)].sort((x,y)=>x-y);return {tp,fp,fn,tn,dice:div(2*tp,2*tp+fp+fn),iou:div(tp,tp+fp+fn),precision:div(tp,tp+fp),recall:div(tp,tp+fn),specificity:div(tn,tn+fp),accuracy:div(tp+tn,a.length),balancedAccuracy:(div(tp,tp+fn)!=null&&div(tn,tn+fp)!=null)?(div(tp,tp+fn)+div(tn,tn+fp))/2:null,hd:d.length?d[d.length-1]:null,hd95:d.length?quantile(d,.95):null,assd:d.length?mean(d):null,surfaceSampled:sampled,surfacePoints:[A.length,B.length]};}
OML.segmentationMetrics=function(actual,pred,opts={}){if(actual.width!==pred.width||actual.height!==pred.height)throw new Error('Reference and prediction masks must have identical dimensions.');const classes=unique([...actual.classes,...pred.classes]).map(Number).sort((a,b)=>a-b);const includeBackground=!!opts.includeBackground,rows=[];for(const c of classes){if(!includeBackground&&c===0)continue;rows.push({class:c,...binaryMaskStats(actual.labels,pred.labels,c,actual.width,actual.height,opts.spacingX||1,opts.spacingY||1)});}const keys=['dice','iou','precision','recall','specificity','accuracy','balancedAccuracy','hd95','assd'];const macro={};for(const k of keys){const v=rows.map(r=>r[k]).filter(Number.isFinite);macro[k]=v.length?mean(v):null;}return {width:actual.width,height:actual.height,classes,perClass:rows,macro,includeBackground};};

function grayFromImageData(img){const n=img.width*img.height,a=new Float64Array(n);for(let i=0;i<n;i++)a[i]=.2126*img.data[4*i]+.7152*img.data[4*i+1]+.0722*img.data[4*i+2];return a;}
function nmi(a,b,bins=64){let amin=Infinity,amax=-Infinity,bmin=Infinity,bmax=-Infinity;for(let i=0;i<a.length;i++){amin=Math.min(amin,a[i]);amax=Math.max(amax,a[i]);bmin=Math.min(bmin,b[i]);bmax=Math.max(bmax,b[i]);}const ha=Array(bins).fill(0),hb=Array(bins).fill(0),joint=Array.from({length:bins},()=>Array(bins).fill(0));for(let i=0;i<a.length;i++){const ia=Math.min(bins-1,Math.floor((a[i]-amin)/(amax-amin+EPS)*bins)),ib=Math.min(bins-1,Math.floor((b[i]-bmin)/(bmax-bmin+EPS)*bins));ha[ia]++;hb[ib]++;joint[ia][ib]++;}function entropy(h,total){let z=0;for(const c of h)if(c){const p=c/total;z-=p*Math.log(p);}return z;}const H1=entropy(ha,a.length),H2=entropy(hb,a.length),flat=joint.flat(),H12=entropy(flat,a.length);return H12>EPS?(H1+H2)/H12:null;}
function ssimUniform(a,b,w,h,r=3){const C1=(.01*255)**2,C2=(.03*255)**2;let sum=0,count=0;for(let y=r;y<h-r;y+=1)for(let x=r;x<w-r;x+=1){let n=0,ma=0,mb=0;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const i=(y+dy)*w+x+dx;ma+=a[i];mb+=b[i];n++;}ma/=n;mb/=n;let va=0,vb=0,cov=0;for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const i=(y+dy)*w+x+dx,da=a[i]-ma,db=b[i]-mb;va+=da*da;vb+=db*db;cov+=da*db;}va/=n-1;vb/=n-1;cov/=n-1;sum+=((2*ma*mb+C1)*(2*cov+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2));count++;}return count?sum/count:null;}
OML.registrationMetrics=function(fixedImg,registeredImg,movingImg=null){if(fixedImg.width!==registeredImg.width||fixedImg.height!==registeredImg.height)throw new Error('Fixed and registered images must have identical dimensions.');const a=grayFromImageData(fixedImg),b=grayFromImageData(registeredImg);let mse=0,ma=0,mb=0;for(let i=0;i<a.length;i++){const d=b[i]-a[i];mse+=d*d;ma+=a[i];mb+=b[i];}mse/=a.length;ma/=a.length;mb/=a.length;let num=0,da=0,db=0,rmsa=0;for(let i=0;i<a.length;i++){const x=a[i]-ma,y=b[i]-mb;num+=x*y;da+=x*x;db+=y*y;rmsa+=a[i]*a[i];}const rmse=Math.sqrt(mse),ncc=da>0&&db>0?num/Math.sqrt(da*db):null,nrmse=rmsa>0?rmse/Math.sqrt(rmsa/a.length):null,psnr=mse>0?20*Math.log10(255)-10*Math.log10(mse):Infinity;const out={mse,rmse,nrmse,psnr,ssim:ssimUniform(a,b,fixedImg.width,fixedImg.height,3),ncc,nmi:nmi(a,b,64),fixed:a,registered:b};if(movingImg){if(movingImg.width!==fixedImg.width||movingImg.height!==fixedImg.height)throw new Error('Moving image must match the fixed image dimensions.');const before=OML.registrationMetrics(fixedImg,movingImg,null);out.before=before;out.improvement={rmse:before.rmse-rmse,ssim:(out.ssim??0)-(before.ssim??0),ncc:(out.ncc??0)-(before.ncc??0),nmi:(out.nmi??0)-(before.nmi??0)};}return out;};
OML.landmarkTRE=function(rows,spacingX=1,spacingY=1){const d=[];for(const r of rows){const fx=+r.fixed_x,fy=+r.fixed_y,px=+(r.registered_x??r.pred_x??r.moving_x),py=+(r.registered_y??r.pred_y??r.moving_y);if([fx,fy,px,py].every(Number.isFinite))d.push(Math.hypot((px-fx)*spacingX,(py-fy)*spacingY));}if(!d.length)return null;return {n:d.length,mean:mean(d),median:quantile(d,.5),rmse:Math.sqrt(mean(d.map(x=>x*x))),p95:quantile(d,.95),max:Math.max(...d),distances:d};};
OML.utils={mean,quantile,variance,correlation,unique,grayFromImageData};
global.OpenMetricLab=OML;
})(window);
