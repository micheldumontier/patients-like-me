// Override QLever UI's IRI link rendering to browse entities within the UI.
//
// Monkey-patches getFormattedResultEntry so that IRIs in query results
// become clickable links that open a SELECT ?p ?o query in the UI.

(function() {
  var _orig = getFormattedResultEntry;

  getFormattedResultEntry = function(str, maxLength, column) {
    var result = _orig(str, maxLength, column);

    // Check if the raw value is an IRI (starts with <http)
    if (str && str.charAt(0) === '<' && str.indexOf('http') === 1) {
      var iri = str.slice(1, -1); // strip < >
      var query = 'SELECT ?p ?o WHERE {\n  <' + iri + '> ?p ?o .\n}';
      var browseUrl = '/' + SLUG + '/?query=' + encodeURIComponent(query);
      var display = result[0];
      result[0] = '<a href="' + browseUrl + '" title="' + iri + '">'
                + '<i class="glyphicon glyphicon-eye-open"></i> '
                + display + '</a>';
    }

    return result;
  };
})();
