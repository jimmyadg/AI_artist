var viewAngle = 75;
var aspectRatio = window.innerWidth / window.innerHeight;
var near = 0.1;
var far = 5000;
var camera = new THREE.PerspectiveCamera(viewAngle, aspectRatio, near, far);
var scene = new THREE.Scene();
var renderer = new THREE.WebGLRenderer();
var container;
var mouseX = 0;
var mouseY = 0;
var fov=0;
var plane;
var mouseX = 0, mouseY = 0;
var windowHalfX = window.innerWidth /2;
var windowHalfY = window.innerHeight /2;
//var loader = new THREE.ImageLoader();

function init(){
  document.addEventListener('mousemove',onMouseMove,false);
  //document.addEventListener( 'mousewheel', onDocumentMouseWheel, false );
  window.addEventListener('resize',onWindowResize,false);
  container = document.getElementById( 'canvas' );
  document.body.appendChild( container );
  renderer.setSize(window.innerWidth, window.innerHeight);
  console.log('jyao');
  container.appendChild( renderer.domElement );
  loadImg();

  var spritey = textRender("hello, ",{
    fontsize:24,
    borderColor:{r:255,g:0,b:0,a:1.0},
    backgroundColor:{r:255,g:100,b:100,a:1.0}
  });
  spritey.position.set(-85,105,55);
  scene.add(spritey);

  camera.position.z = 20;
  camera.lookAt(new THREE.Vector3(0,0,0));
  scene.add(camera);
}

function loadImg(){
  var loader = new THREE.TextureLoader();
  loader.load('./img/test.jpg',function(texture){
    var geometry = new THREE.PlaneGeometry( 30, 20, 32 );
    var material = new THREE.MeshBasicMaterial({
      map:texture,
      overdraw: 0.5
    });
    plane = new THREE.Mesh(geometry,material);
    scene.add(plane);
  });
}

function textRender(message, parameters){
  if(parameters === undefined) parameters ={};
  var fontface = parameters.hasOwnProperty("fontface")?parameters["fontface"]:"Roboto";
  var fontsize = parameters.hasOwnProperty("fontsize")?parameters["fontsize"]:18;
  var borderThickness = parameters.hasOwnProperty("borderThickness")?parameters["borderThickness"]:4;
  var borderColor = parameters.hasOwnProperty("borderColor")?parameters["borderColor"]:{r:0,g:0,b:0,a:1.0};
  var backgroundColor = parameters.hasOwnProperty("backgroundColor")?parameters["backgroundColor"]:{r:255,g:255,b:255,a:1.0};
  //var spriteAlignment = THREE.SpriteAlignment.topLeft;
  var canvas = document.createElement('canvas');
  var context = canvas.getContext('2d');
  context.font = "Bold" + fontsize + "px "+ fontface;
  var metrics = context.measureText(message);
  var textWidth = metrics.width;
  //background color
  context.fillStyle = "rgba(" + backgroundColor.r +"."+backgroundColor.g+","+backgroundColor.b+","+backgroundColor.a+")";
  //border color
  context.strokeStyle = "rgba(" + backgroundColor.r +"."+backgroundColor.g+","+backgroundColor.b+","+backgroundColor.a+")";
  context.lineWidth = borderThickness;
  roundRect(context,borderThickness/2,borderThickness/2,textWidth+borderThickness,fontsize*1.4+borderThickness,6);

  //text color
  context.fillStyle="rgba(0,0,0,1.0)";
  context.fillText(message,borderThickness,fontsize+borderThickness);
  var texture = new THREE.Texture(canvas);
  texture.needsUpdate = true;
  var spriteMaterial = new THREE.spriteMaterial({
    map:texture,
    useScreenCoordinates: false
    //alignment: spriteAlignment
  });
  var sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(100,50,1.0);
  return sprite;
}

//drawing rectangles
function roundRect(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y);
  ctx.lineTo(x+w-r,y);
  ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r);
  ctx.quadraticCurveTo(x+w,y+h,x,y+h-r);
  ctx.lineTo(x,y+r);
  ctx.quadraticCurveTo(x,y,x+r,y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}


function animatedRender(){
  requestAnimationFrame(animatedRender);
  renderer.render(scene,camera);

  //set background color
  bgColor = new THREE.Color(255,255,255);
  renderer.setClearColor(bgColor,1);
  camera.position.x += ( mouseX/8 - camera.position.x *30) * 0.05;
  camera.position.y += ( - mouseY/8 - camera.position.y*30 ) * 0.05;
  camera.lookAt( scene.position );

}

function onMouseMove(event){
  mouseX = event.clientX - windowHalfX;
  mouseY = event.clientY - windowHalfY;
}

function onDocumentMouseWheel(event)
{
  camera.position.z -= event.wheelDeltaY * 0.5;
}


function onWindowResize(){
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}

init();
animatedRender();
