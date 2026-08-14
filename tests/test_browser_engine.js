global.window=global;
require('../web/openmetriclab-engine.js');
const E=global.OpenMetricLab;
function close(a,b,t=1e-10){if(a==null||Math.abs(a-b)>t)throw new Error(`${a} != ${b}`);}
if(E.engine.externalMetricLibraries.length!==0)throw new Error('browser engine should not declare metric dependencies');
let r=E.regressionMetrics([1,2,3],[2,2,4]);
close(r.mae,2/3); close(r.r2,0);
let c=E.classificationMetrics([0,1,0,1],[0,1,0,1],[[.8,.2],[.2,.8],[.5,.5],[.5,.5]],[0,1]);
close(c.rocAuc,.875);
console.log('PASS native browser metric engine · regression + tied-score ROC');
