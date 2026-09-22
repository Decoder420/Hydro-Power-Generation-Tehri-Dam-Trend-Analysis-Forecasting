export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Redirect /portal or /portal/ to root
    if (url.pathname === '/portal' || url.pathname === '/portal/') {
      return Response.redirect(`${url.origin}/`, 301);
    }

    // Serve static assets from ./portal
    return env.ASSETS.fetch(request);
  },
};
