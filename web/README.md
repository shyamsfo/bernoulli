# web/

Source of the [bernoulli.live](https://www.bernoulli.live/) landing page.

```
web/
├── index.html    # single-page landing (inlined CSS + JS, no build step)
└── favicon.svg
```

Deploy from this directory to `ssd2:/var/www/bernoulli.live/html/` with:

```bash
just deploy-web
```

The recipe is a plain `rsync -av --delete --rsync-path="sudo rsync" ...`. Needs passwordless `sudo` for your user on `ssd2`. No build pipeline — edit, deploy, done.

If you prefer to iterate visually, open `web/index.html` directly in a browser or load it into Claude Desktop as an artifact.
