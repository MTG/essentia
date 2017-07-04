var $headerlinks = jQuery('.headerlink');
var $body = jQuery('body');

$body.addClass('essentia-docs-body');

[].slice.call($headerlinks).forEach(function replaceHeaderLinkContent($el) {
  $el.innerHTML = '<span class="glyphicon glyphicon-link" aria-hidden="true"></span>';
});
