// File: custom.js

/**
 * This function will make sure all links in the document open as separate tabs
 * instead of opening on the current user window.
 */
window.onload = function () {
    var links = document.getElementsByTagName('a');
    for (var i = 0; i < links.length; i++) {
        links[i].target = '_blank'; // Open links in a new tab
    }
};
