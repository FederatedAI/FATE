const path = require('path');
const transformMarkdown = require('markdown-magic');
const badgePlugin = require('markdown-magic-branch-badge');

const config = {
  transforms: {
    badgePlugin,
  },
};

function callback() {
  console.log('ReadME generated.');
}

const markdownPath = path.join(__dirname, 'README.md');
transformMarkdown(markdownPath, config, callback);
