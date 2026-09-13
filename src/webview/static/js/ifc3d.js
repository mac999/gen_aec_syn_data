/* IFC 3D view: server-tessellated meshes drawn with three.js.
   Orbit/pan/zoom is implemented here so no example bundle is needed. */
import * as THREE from "three";

let renderer, scene, camera, host, frame;
const orbit = { target: new THREE.Vector3(), radius: 10, theta: 0.9, phi: 1.05 };
// 렌더 모드와 투명도는 재생성 없이 바꿔야 하므로 메시 참조를 들고 있는다
let meshes = [];
let axes = null;
let state = { mode: 'shaded', opacity: 1 };
let home = null;

function themeColour() {
  const style = getComputedStyle(document.documentElement);
  return (style.getPropertyValue("--bg") || "#12151a").trim();
}

function dispose() {
  meshes = [];
  axes = null;
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
    const colour = new THREE.Color(group.colour);
    const material = new THREE.MeshLambertMaterial({
      color: colour, side: THREE.DoubleSide,
      transparent: true, opacity: 1, depthWrite: true
    });
    const mesh = new THREE.Mesh(geometry, material);
    // 와이어프레임은 같은 지오메트리의 모서리만 그린다. 면 대각선을 빼려고
    // EdgesGeometry를 쓰며, 임계각 이하의 매끈한 이음매는 선을 만들지 않는다
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 25),
      new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 1 })
    );
    edges.visible = false;
    scene.add(mesh);
    scene.add(edges);
    meshes.push({ mesh, edges });
  });

  const grid = new THREE.GridHelper(span * 3, 30, 0x445066, 0x2a313d);
  grid.position.y = -(size[1] || 0) / 2;
  scene.add(grid);

  // XYZ 축: 모델 바닥 모서리에 두고 모델 크기에 비례시킨다.
  // 화면 중앙을 가리지 않도록 격자 원점이 아니라 모델 경계에 붙인다
  axes = new THREE.Group();
  const len = span * 0.42;
  const AX = [
    { dir: [1, 0, 0], colour: 0xd45b4a, label: "X" },
    { dir: [0, 1, 0], colour: 0x63b463, label: "Y" },
    { dir: [0, 0, 1], colour: 0x4a86d4, label: "Z" }
  ];
  AX.forEach(a => {
    const v = new THREE.Vector3(...a.dir).multiplyScalar(len);
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), v]),
      new THREE.LineBasicMaterial({ color: a.colour })
    );
    axes.add(line);
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(len * 0.035, len * 0.11, 12),
      new THREE.MeshBasicMaterial({ color: a.colour })
    );
    cone.position.copy(v);
    if (a.label === "X") cone.rotation.z = -Math.PI / 2;
    if (a.label === "Z") cone.rotation.x = Math.PI / 2;
    axes.add(cone);
    axes.add(makeLabel(a.label, a.colour,
      v.clone().multiplyScalar(1.13), len * 0.16));
  });
  axes.position.set(-(size[0] || 0) / 2 - len * 0.25,
                    -(size[1] || 0) / 2,
                    -(size[2] || 0) / 2 - len * 0.25);
  scene.add(axes);

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

function makeLabel(text, colour, position, scale) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#" + new THREE.Color(colour).getHexString();
  ctx.font = "bold 44px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 32, 34);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(c), transparent: true, depthTest: false
  }));
  sprite.position.copy(position);
  sprite.scale.setScalar(scale);
  return sprite;
}

function setMode(mode) {
  state.mode = mode;
  meshes.forEach(({ mesh, edges }) => {
    // wireframe: 면을 숨기고 모서리만. shaded+edges: 둘 다.
    mesh.visible = mode !== "wireframe";
    edges.visible = mode === "wireframe" || mode === "shaded_edges";
    mesh.material.flatShading = mode === "flat";
    mesh.material.needsUpdate = true;
  });
}

function setOpacity(value) {
  state.opacity = value;
  meshes.forEach(({ mesh, edges }) => {
    mesh.material.opacity = value;
    // 반투명일 때 깊이 기록을 끄지 않으면 뒤쪽 면이 가려져 속이 안 보인다
    mesh.material.depthWrite = value >= 0.99;
    edges.material.opacity = Math.min(1, value + 0.35);
  });
}

function setAxes(visible) {
  if (axes) axes.visible = visible;
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
  show(mount, data) {
    show(mount, data);
    observer.observe(mount);
    setMode(state.mode);
    setOpacity(state.opacity);
  },
  dispose, reset, retheme, setMode, setOpacity, setAxes
};
