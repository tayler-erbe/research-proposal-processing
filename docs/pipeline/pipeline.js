/* Headline tabs for the pipeline brief. Content only, no data calls. */
'use strict';
(function () {
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.hero-tabs button'));
  function show(n) {
    tabs.forEach(function (b, i) {
      b.setAttribute('aria-selected', i === n);
      b.tabIndex = i === n ? 0 : -1;
      document.getElementById(b.getAttribute('aria-controls')).hidden = i !== n;
    });
  }
  tabs.forEach(function (b, i) {
    b.addEventListener('click', function () { show(i); });
    b.addEventListener('keydown', function (e) {
      if (['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].indexOf(e.key) < 0) return;
      e.preventDefault();
      var n = (i + (e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : -1) + tabs.length) % tabs.length;
      show(n); tabs[n].focus();
    });
  });
})();
