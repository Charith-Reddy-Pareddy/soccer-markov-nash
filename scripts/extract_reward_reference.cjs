const fs=require('fs'),vm=require('vm'),crypto=require('crypto');
// Inputs are the inspected public viewer and data files linked in docs/reward-q-audit.md.
const [viewerPath,dataPath,outputPath]=process.argv.slice(2);
if(!outputPath) throw new Error('Usage: node scripts/extract_reward_reference.cjs viewer.js data.js output.json');
const viewer=fs.readFileSync(viewerPath,'utf8');
const data=fs.readFileSync(dataPath,'utf8');
const ctx={window:{}};vm.createContext(ctx);vm.runInContext(data,ctx);
vm.runInContext(viewer.slice(viewer.indexOf('  var GAMMA'),viewer.indexOf('  function dist')),ctx);
vm.runInContext(`var states=[],transitions=[];
for(var x0=0;x0<7;x0++)for(var y0=0;y0<5;y0++)for(var x1=0;x1<7;x1++)for(var y1=0;y1<5;y1++)for(var b=0;b<2;b++){
if(x0===x1 && y0===y1)continue;
var s=[x0,y0,x1,y1,b],pairs=[];states.push(s);
for(var a=0;a<4;a++)for(var o=0;o<4;o++){var r=stepDet(s,a,o);pairs.push({ns:r.ns,reward:reward(r.ns,r.w)});}transitions.push(pairs);
}`,ctx);
fs.writeFileSync(outputPath,JSON.stringify({states:ctx.states,values:ctx.window.SOCCER_DET.V,transitions:ctx.transitions}));
console.log({viewer_sha256:crypto.createHash('sha256').update(viewer).digest('hex'),data_sha256:crypto.createHash('sha256').update(data).digest('hex'),states:ctx.states.length});
