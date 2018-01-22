var $headerlinks = $('.headerlink');
var headerLinkContent = '<span class="glyphicon glyphicon-link" aria-hidden="true"></span>';

[].slice.call($headerlinks).forEach(function replaceHeaderLinkContent(el) {
  el.innerHTML = headerLinkContent;
});

var $sectionWithHeaders = $('main section[id]>:header');
[].slice.call($sectionWithHeaders).forEach(function addHeaderLink(el) {
  var parent = el.parentElement;
  var parentId = parent.id;
  var headerLink = document.createElement('a');
  headerLink.classList.add('headerlink');
  headerLink.href = '#' + parentId;
  headerLink.title = 'Permalink to this headline';
  headerLink.innerHTML = headerLinkContent;
  el.appendChild(headerLink);
});
