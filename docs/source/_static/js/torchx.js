/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


function setLinks(colabLink, notebookLink, githubLink) {
    $("#torchx-google-colab-link").wrap("<a href='" + colabLink + "' data-behavior='call-to-action-event' data-response='Run in Google Colab' target='_blank'/>");
    $("#torchx-download-notebook-link").wrap("<a href='" + notebookLink + "' data-behavior='call-to-action-event' data-response='Download Notebook'/>");
    $("#torchx-github-view-link").wrap("<a href='" + githubLink + "' data-behavior='call-to-action-event' data-response='View on Github' target='_blank'/>");

    setTimeout(() => {
        $(".pytorch-call-to-action-links").show();
    }, 0);
}

const version = $(".version").text().trim().split(" ")[0].substr(1);
const colabBase = "https://colab.research.google.com/github/pytorch/torchx/blob/gh-pages/" + version
const githubBase = "https://github.com/pytorch/torchx/blob/main/"

var downloadNote = $(".sphx-glr-download-link-note.admonition.note");
var isNBSphinx = $("#is-nbsphinx");

if (downloadNote.length >= 1) {
    var tutorialUrlArray = $("#tutorial-type").text().split('/');
        tutorialUrlArray[0] = tutorialUrlArray[0].replace("examples_", "")

    var githubLink = githubBase + "examples/" + tutorialUrlArray.join("/") + ".py",
        notebookLink = $(".reference.download")[1].href,
        notebookDownloadPath = notebookLink.split('_downloads')[1],
        colabLink = colabBase + "/_downloads" + notebookDownloadPath;

    setLinks(colabLink, notebookLink, githubLink);
} else if (isNBSphinx.length >= 1) {
    const notebookLink = window.location.pathname.replace(".html", ".ipynb");
    const colabLink = colabBase + notebookLink;
    const githubLink = githubBase + "docs/source" + window.location.pathname.replace(".html", ".md");

    setLinks(colabLink, notebookLink, githubLink);
} else {
    console.log("hiding")
    $(".pytorch-call-to-action-links").hide();
}
