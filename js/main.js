var submitBt = document.getElementById('submit');
submitBt.onclick = function(){
  var script = document.createElement('script');
  script.type = "text/javascript";
  script.src = "js/script.js";
  document.body.appendChild(script);
  submitBt.style.visibility = "hidden";
  document.getElementById('info').style.visibility = 'visible';
  return false;
}
