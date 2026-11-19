// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Assemble docs/page/strata.html from strata.src.html, the shared CSS and chart helpers,
//! docs/data/strata.csv (generation 1) and docs/data/strata-gen2.csv (generation 2). Run from
//! the repository root: `zig build page`.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const dir = std.Io.Dir.cwd();
    const limit: std.Io.Limit = .limited(64 << 20);
    var page = try dir.readFileAlloc(io, "docs/page/strata.src.html", gpa, limit);
    defer gpa.free(page);
    const parts = [_]struct { key: []const u8, path: []const u8 }{
        .{ .key = "__CSS__", .path = "docs/page/lib.css.html" },
        .{ .key = "__CHARTS__", .path = "docs/page/charts.js" },
        .{ .key = "__CSV__", .path = "docs/data/strata.csv" },
        .{ .key = "__CSV2__", .path = "docs/data/strata-gen2.csv" },
    };
    for (parts) |p| {
        const text = try dir.readFileAlloc(io, p.path, gpa, limit);
        defer gpa.free(text);
        const next = try std.mem.replaceOwned(u8, gpa, page, p.key, std.mem.trim(u8, text, "\n"));
        gpa.free(page);
        page = next;
    }
    try dir.writeFile(io, .{ .sub_path = "docs/page/strata.html", .data = page });
}
