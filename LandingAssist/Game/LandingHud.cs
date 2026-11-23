// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.Collections.Generic;
using UnityEngine;

namespace MaydayLanding;

public enum HudTone { Monitor, Guide, Armed, Engaged, Alert }

/// <summary>What the panel shows; built by the mod each frame, drawn by <see cref="LandingHud"/>.</summary>
public sealed class HudModel
{
    public string Mode = "监测";
    public HudTone ModeTone = HudTone.Monitor;
    public string Phase = "";
    public string Instruction = "";
    public HudTone InstructionTone = HudTone.Monitor;
    public readonly List<(string Label, string Value, bool Warn)> Metrics = new(4);
    public string Detail = "";
    public string Controls = "";
    public string Recording = "";
    public bool RecordingActive;
}

/// <summary>Game-style panel under the native aircraft card (top left). Sizes follow screen height,
/// matched to a 1125-px-high capture: the aircraft card spans x 25-268, y 25-298 and uses a
/// translucent slate fill (RGB 25,40,55 at ~62%) with a green accent.</summary>
public sealed class LandingHud : IDisposable
{
    private static readonly Color Fill = new(25 / 255f, 40 / 255f, 55 / 255f, 0.64f);
    private static readonly Color Divider = new(1f, 1f, 1f, 0.12f);
    private static readonly Color Label = new(0.78f, 0.83f, 0.87f);
    private static readonly Color Value = new(0.97f, 0.98f, 1f);
    private static readonly Color Muted = new(0.66f, 0.72f, 0.77f);
    private Texture2D? _rounded, _plain;
    private GUIStyle? _background;
    private Font? _font;
    private GUIStyle? _text;
    private float _scale = -1f;
    private readonly Dictionary<(int, bool, TextAnchor, bool), GUIStyle> _styles = new();

    public static Color ToneColor(HudTone tone) => tone switch
    {
        HudTone.Engaged => new Color(123 / 255f, 194 / 255f, 77 / 255f),
        HudTone.Armed => new Color(242 / 255f, 184 / 255f, 75 / 255f),
        HudTone.Guide => new Color(150 / 255f, 205 / 255f, 240 / 255f),
        HudTone.Alert => new Color(240 / 255f, 122 / 255f, 106 / 255f),
        _ => Value
    };

    public void Draw(HudModel model)
    {
        float u = Screen.height / 1125f;
        Prepare(u);
        float pad = 16f * u, width = 243f * u, inner = width - 2 * pad;
        float x = 25f * u, y = 310f * u;

        // Measure first so the background fits the content.
        float h = pad;
        h += Line(17, false, model.Mode, inner, measure: true) + 4f * u;
        if (model.Phase.Length > 0) h += Line(28, true, model.Phase, inner, measure: true) + 2f * u;
        if (model.Instruction.Length > 0) h += Line(16, false, model.Instruction, inner, measure: true) + 8f * u;
        int rows = (model.Metrics.Count + 1) / 2;
        if (rows > 0) h += 9f * u + rows * 48f * u;
        if (model.Detail.Length > 0) h += Line(13, false, model.Detail, inner, measure: true) + 4f * u;
        h += 9f * u + Line(13, false, model.Controls, inner, measure: true);
        if (model.Recording.Length > 0) h += Line(13, false, model.Recording, inner, measure: true);
        h += pad;

        var previous = GUI.color;
        GUI.color = Fill;
        GUI.Box(new Rect(x, y, width, h), GUIContent.none, Background());
        GUI.color = previous;

        float cy = y + pad;
        // Header: section label left, mode right (like "飞机 ... 100%").
        float headH = Line(17, false, "进近辅助", inner, measure: true);
        Text(new Rect(x + pad, cy, inner, headH), "进近辅助", 17, false, Label, TextAnchor.UpperLeft);
        Text(new Rect(x + pad, cy, inner, headH), model.Mode, 17, true, ToneColor(model.ModeTone), TextAnchor.UpperRight);
        cy += headH + 4f * u;
        if (model.Phase.Length > 0)
        {
            float ph = Line(28, true, model.Phase, inner, measure: true);
            Text(new Rect(x + pad, cy, inner, ph), model.Phase, 28, true, Value, TextAnchor.UpperLeft);
            cy += ph + 2f * u;
        }
        if (model.Instruction.Length > 0)
        {
            float ih = Line(16, false, model.Instruction, inner, measure: true);
            Text(new Rect(x + pad, cy, inner, ih), model.Instruction, 16, false, ToneColor(model.InstructionTone), TextAnchor.UpperLeft, wrap: true);
            cy += ih + 8f * u;
        }
        if (rows > 0)
        {
            Rule(x + pad, cy, inner);
            cy += 9f * u;
            float col = inner / 2f;
            for (int i = 0; i < model.Metrics.Count; i++)
            {
                var (label, value, warn) = model.Metrics[i];
                float mx = x + pad + (i % 2) * col, my = cy + (i / 2) * 48f * u;
                Text(new Rect(mx, my, col, 18f * u), label, 13, false, Muted, TextAnchor.UpperLeft);
                Text(new Rect(mx, my + 17f * u, col, 28f * u), value, 21, true, warn ? ToneColor(HudTone.Alert) : Value, TextAnchor.UpperLeft);
            }
            cy += rows * 48f * u;
        }
        if (model.Detail.Length > 0)
        {
            float dh = Line(13, false, model.Detail, inner, measure: true);
            Text(new Rect(x + pad, cy, inner, dh), model.Detail, 13, false, Label, TextAnchor.UpperLeft, wrap: true);
            cy += dh + 4f * u;
        }
        Rule(x + pad, cy, inner);
        cy += 9f * u;
        float ch = Line(13, false, model.Controls, inner, measure: true);
        Text(new Rect(x + pad, cy, inner, ch), model.Controls, 13, false, Muted, TextAnchor.UpperLeft, wrap: true);
        cy += ch;
        if (model.Recording.Length > 0)
            Text(new Rect(x + pad, cy, inner, Line(13, false, model.Recording, inner, measure: true)), model.Recording, 13, false,
                model.RecordingActive ? ToneColor(HudTone.Alert) : Muted, TextAnchor.UpperLeft, wrap: true);
    }

    private void Prepare(float u)
    {
        if (_text != null && Math.Abs(_scale - u) < 1e-4f) return;
        _scale = u;
        if (_font == null)
        {
            try { _font = Font.CreateDynamicFontFromOSFont(new[] { "Microsoft YaHei UI", "Microsoft YaHei", "Arial" }, 17); }
            catch (Exception) { /* The default GUI font remains available. */ }
        }
        _styles.Clear();
        _text = new GUIStyle(GUI.skin.label) { richText = false, clipping = TextClipping.Overflow, padding = new RectOffset(0, 0, 0, 0) };
        if (_font != null) _text.font = _font;
    }

    private float Line(int size, bool bold, string text, float width, bool measure)
    {
        var style = Style(size, bold, Value, TextAnchor.UpperLeft, wrap: true);
        return style.CalcHeight(new GUIContent(text), width);
    }

    private void Text(Rect rect, string text, int size, bool bold, Color color, TextAnchor anchor, bool wrap = false)
    {
        var style = Style(size, bold, color, anchor, wrap);
        GUI.Label(rect, text, style);
    }

    private GUIStyle Style(int size, bool bold, Color color, TextAnchor anchor, bool wrap)
    {
        var key = (size, bold, anchor, wrap);
        if (!_styles.TryGetValue(key, out var style))
        {
            style = new GUIStyle(_text!)
            {
                fontSize = Mathf.Max(9, Mathf.RoundToInt(size * _scale)),
                fontStyle = bold ? FontStyle.Bold : FontStyle.Normal,
                alignment = anchor, wordWrap = wrap
            };
            _styles[key] = style;
        }
        style.normal.textColor = color;
        return style;
    }

    private void Rule(float x, float y, float width)
    {
        _plain ??= Solid();
        var previous = GUI.color;
        GUI.color = Divider;
        GUI.DrawTexture(new Rect(x, y, width, Mathf.Max(1f, _scale)), _plain);
        GUI.color = previous;
    }

    private GUIStyle Background()
    {
        _rounded ??= Rounded(64, 14);
        return _background ??= new GUIStyle { normal = { background = _rounded }, border = new RectOffset(16, 16, 16, 16) };
    }

    private static Texture2D Solid()
    {
        var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false) { hideFlags = HideFlags.HideAndDontSave };
        texture.SetPixel(0, 0, Color.white);
        texture.Apply();
        return texture;
    }

    /// <summary>White rounded square with anti-aliased corners; GUI.color tints it.</summary>
    private static Texture2D Rounded(int size, float radius)
    {
        var texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
            { hideFlags = HideFlags.HideAndDontSave, wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        for (int py = 0; py < size; py++)
            for (int px = 0; px < size; px++)
            {
                float dx = Mathf.Max(0f, Mathf.Max(radius - (px + 0.5f), (px + 0.5f) - (size - radius)));
                float dy = Mathf.Max(0f, Mathf.Max(radius - (py + 0.5f), (py + 0.5f) - (size - radius)));
                float alpha = Mathf.Clamp01(radius - Mathf.Sqrt(dx * dx + dy * dy) + 0.5f);
                texture.SetPixel(px, py, new Color(1f, 1f, 1f, alpha));
            }
        texture.Apply();
        return texture;
    }

    public void Dispose()
    {
        if (_rounded != null) UnityEngine.Object.Destroy(_rounded);
        if (_plain != null) UnityEngine.Object.Destroy(_plain);
        if (_font != null) UnityEngine.Object.Destroy(_font);
        _rounded = _plain = null;
        _font = null;
        _text = null;
        _background = null;
        _styles.Clear();
    }
}
