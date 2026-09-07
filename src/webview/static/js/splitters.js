/* Draggable panel splitters. Sizes are CSS variables on :root, kept in
   localStorage so a tuned layout survives a reload. */
(function () {
  "use strict";

  const KEY = "aec.layout";
  const root = document.documentElement;

  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { saved = {}; }
  Object.keys(saved).forEach(name => root.style.setProperty(name, saved[name]));

  function persist(name) {
    saved[name] = root.style.getPropertyValue(name);
    try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { /* private mode */ }
  }

  function drag(handle, axis, name, measure) {
    handle.addEventListener("mousedown", down => {
      down.preventDefault();
      handle.classList.add("active");
      document.body.classList.add("dragging", axis === "x" ? "dragging-col" : "dragging-row");

      const move = ev => root.style.setProperty(name, measure(ev));
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        handle.classList.remove("active");
        document.body.classList.remove("dragging", "dragging-col", "dragging-row");
        persist(name);
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });

    // Double-click restores the default width/height.
    handle.addEventListener("dblclick", () => {
      root.style.removeProperty(name);
      delete saved[name];
      try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { /* ignore */ }
    });
  }

  function init() {
    const layout = document.querySelector(".layout");
    if (!layout) return;

    document.querySelectorAll(".vsplit").forEach(handle => {
      const side = handle.dataset.split;                 // "left" | "right"
      const name = "--w-" + side;
      drag(handle, "x", name, ev => {
        const box = layout.getBoundingClientRect();
        const px = side === "left" ? ev.clientX - box.left : box.right - ev.clientX;
        const other = document.querySelector(".col-" + (side === "left" ? "right" : "left"))
          .getBoundingClientRect().width;
        const max = box.width - other - 280;             // keep the centre usable
        return (Math.max(170, Math.min(px, max)) / box.width * 100).toFixed(2) + "%";
      });
    });

    document.querySelectorAll(".hsplit").forEach(handle => {
      const side = handle.dataset.split;
      const name = "--h-" + side;
      drag(handle, "y", name, ev => {
        const box = handle.parentElement.getBoundingClientRect();
        const px = Math.max(90, Math.min(ev.clientY - box.top, box.height - 110));
        return (px / box.height * 100).toFixed(2) + "%";
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
