/* IFC 3D view: server-tessellated meshes drawn with three.js.
   Orbit/pan/zoom is implemented here so no example bundle is needed. */
import * as THREE from "three";

let renderer, scene, camera, host, frame;
const orbit = { target: new THREE.Vector3(), radius: 10, theta: 0.9, phi: 1.05 };
let home = null;

function themeColour() {
  const style = getComputedStyle(document.documentElement);
  return (style.getPropertyValue("--bg") || "#12151a").trim();
}

function dispose() {
  if (frame) cancelAnimationFrame(frame);
  frame = null;
  if (renderer) {
    renderer.dispose();
    if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
  }
  if (scene) {
    scene.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
  }
  renderer = scene = camera = host = null;
}

function place() {
  const sinPhi = Math.sin(orbit.phi);
  camera.position.set(
    orbit.target.x + orbit.radius * sinPhi * Math.sin(orbit.theta),
    orbit.target.y + orbit.radius * Math.cos(orbit.phi),
    orbit.target.z + orbit.radius * sinPhi * Math.cos(orbit.theta)
  );
  camera.lookAt(orbit.target);
}

function resize() {
  if (!renderer || !host) return;
  const w = host.clientWidth || 1;
  const h = host.clientHeight || 1;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function controls(dom) {
  let mode = 0, lastX = 0, lastY = 0;

  dom.addEventListener("mousedown", e => {
    mode = e.button === 0 && !e.shiftKey ? 1 : 2;
    lastX = e.clientX; lastY = e.clientY;
    e.preventDefault();
  });
  window.addEventListener("mouseup", () => { mode = 0; });
  window.addEventListener("mousemove", e => {
    if (!mode || !camera) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (mode === 1) {
      orbit.theta -= dx * 0.006;
      orbit.phi = Math.max(0.05, Math.min(Math.PI - 0.05, orbit.phi - dy * 0.006));
    } else {
      // Pan along the camera's own right/up axes.
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      const scale = orbit.radius * 0.0015;
      orbit.target.addScaledVector(right, -dx * scale);
      orbit.target.addScaledVector(up, dy * scale);
    }
    place();
  });
  dom.addEventListener("wheel", e => {
    orbit.radius = Math.max(0.5, orbit.radius * (e.deltaY > 0 ? 1.12 : 0.89));
    place();
    e.preventDefault();
  }, { passive: false });
  dom.addEventListener("contextmenu", e => e.preventDefault());
}

function show(mount, data) {
  dispose();
  host = mount;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(themeColour());

  camera = new THREE.PerspectiveCamera(45, 1, 0.05, 1e6);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  host.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(1, 2, 1.5);
  scene.add(key);

  const size = data.size || [10, 10, 10];
  const span = Math.max(size[0] || 1, size[1] || 1, size[2] || 1);

  (data.groups || []).forEach(group => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position",
      new THREE.Float32BufferAttribute(group.positions, 3));
    geometry.setIndex(group.indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshLambertMaterial({
      color: new THREE.Color(group.colour),
      side: THREE.DoubleSide
    });
    scene.add(new THREE.Mesh(geometry, material));
  });

  const grid = new THREE.GridHelper(span * 3, 30, 0x445066, 0x2a313d);
  grid.position.y = -(size[1] || 0) / 2;
  scene.add(grid);

  orbit.target.set(0, 0, 0);
  orbit.radius = span * 1.9 || 10;
  orbit.theta = 0.9;
  orbit.phi = 1.05;
  home = { radius: orbit.radius, theta: orbit.theta, phi: orbit.phi };
  place();
  resize();
  controls(renderer.domElement);

  const loop = () => {
    frame = requestAnimationFrame(loop);
    if (renderer && scene && camera) renderer.render(scene, camera);
  };
  loop();
}

function reset() {
  if (!camera || !home) return;
  orbit.target.set(0, 0, 0);
  orbit.radius = home.radius;
  orbit.theta = home.theta;
  orbit.phi = home.phi;
  place();
}

function retheme() {
  if (scene) scene.background = new THREE.Color(themeColour());
}

window.addEventListener("resize", resize);
const observer = new ResizeObserver(resize);
window.AECViewer3D = {
  show(mount, data) { show(mount, data); observer.observe(mount); },
  dispose, reset, retheme
};
