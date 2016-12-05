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
var windowHalfX = window.innerWidth / 2;
var windowHalfY = window.innerHeight / 2;
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
  camera.position.z = 50;
  camera.lookAt(new THREE.Vector3(0,0,0));
  scene.add(camera);
}

function loadImg(){
  var loader = new THREE.TextureLoader();
  loader.load('./img/test.png',function(texture){
    var geometry = new THREE.PlaneGeometry( 30, 20, 32 );
    var material = new THREE.MeshBasicMaterial({
      map:texture,
      overdraw: 0.5
    });
    plane = new THREE.Mesh(geometry,material);
    scene.add(plane);
  });
}


  function animatedRender(){
    requestAnimationFrame(animatedRender);
    renderer.render(scene,camera);

    //set background color
    bgColor = new THREE.Color(255,255,255);
    renderer.setClearColor(bgColor,1);
    camera.position.x += (mouseX - camera.position.x) * 0.01;
    camera.position.y += (-mouseY - camera.position.y) * 0.01;

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
