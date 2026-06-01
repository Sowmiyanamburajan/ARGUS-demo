// static/behavior.js
// Simple capture of typing inter-key-intervals and mouse moves
(function(){
  let lastKeyTime = null;
  let ikis = [];
  document.addEventListener('keydown', (e) => {
    const t = Date.now();
    if(lastKeyTime) {
      ikis.push(t - lastKeyTime);
    }
    lastKeyTime = t;
  });

  let mouseMoves = [];
  document.addEventListener('mousemove', (e) => {
    mouseMoves.push({x:e.clientX, y:e.clientY, t: Date.now()});
    if(mouseMoves.length>500) mouseMoves.shift();
  });

  // send sample to server
  window.sendBehaviorSample = async function(profile_id){
    const mean_iki = ikis.length ? ikis.reduce((a,b)=>a+b)/ikis.length : 0;
    const var_iki = ikis.length ? ikis.map(x=>Math.pow(x-mean_iki,2)).reduce((a,b)=>a+b)/ikis.length : 0;
    const mstats = {moves: mouseMoves.length};
    const payload = {profile_id: profile_id || 'guest', typing_stats:{mean_iki, var_iki, count: ikis.length}, mouse_stats: mstats};
    await fetch('/behavior/sample', {method:'POST', body: JSON.stringify(payload), headers:{'Content-Type':'application/json'}});
    // clear
    ikis = []; mouseMoves=[];
  };
})();
