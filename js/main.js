var submitBt = document.getElementById('submit');
submitBt.onclick = function(){
  var script = document.createElement('script');
  script.type = "text/javascript";
  script.src = "js/script.js";
  document.body.appendChild(script);
  submitBt.style.visibility = "hidden";
  return false;
}
