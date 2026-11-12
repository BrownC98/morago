// Reads and presses the controls of an app window through the macOS
// accessibility API, as a screen reader does. Used by check.sh.
//   ax trusted                      exit 0 if this process may use the API
//   ax dump <pid>                   one line per element: role|subrole|enabled|label
//   ax press <pid> <role> <text> [n]  AXPress on the n-th (default 1) element of
//                                   that role ("*" = any) whose label contains text
import ApplicationServices
import Foundation

func attr(_ e: AXUIElement, _ name: String) -> AnyObject? {
    var v: AnyObject?
    return AXUIElementCopyAttributeValue(e, name as CFString, &v) == .success ? v : nil
}

func text(_ e: AXUIElement, _ name: String) -> String {
    guard let v = attr(e, name) else { return "" }
    if let s = v as? String { return s }
    if let n = v as? NSNumber { return n.stringValue }
    return ""
}

func label(_ e: AXUIElement) -> String {
    ["AXTitle", "AXDescription", "AXValue"].map { text(e, $0) }.filter { !$0.isEmpty }
        .joined(separator: " ¦ ").replacingOccurrences(of: "\n", with: " ")
}

/// Depth-first; `visit` returns true to stop.
@discardableResult
func walk(_ e: AXUIElement, _ depth: Int, _ visit: (AXUIElement, Int) -> Bool) -> Bool {
    if visit(e, depth) { return true }
    if depth > 80 { return false }
    for c in (attr(e, "AXChildren") as? [AXUIElement]) ?? [] where walk(c, depth + 1, visit) {
        return true
    }
    return false
}

let args = CommandLine.arguments
guard args.count >= 2 else { exit(2) }
if args[1] == "trusted" {
    let ok = AXIsProcessTrusted()
    print("AXIsProcessTrusted = \(ok)")
    exit(ok ? 0 : 1)
}
guard args.count >= 3, let pid = Int32(args[2]) else { exit(2) }
let app = AXUIElementCreateApplication(pid)
AXUIElementSetMessagingTimeout(app, 10)

switch args[1] {
case "dump":
    walk(app, 0) { e, depth in
        let enabled = (attr(e, "AXEnabled") as? Bool) == false ? "off" : "on"
        print("\(String(repeating: " ", count: depth))\(text(e, "AXRole"))|\(text(e, "AXSubrole"))|\(enabled)|\(label(e))")
        return false
    }
case "press":
    guard args.count >= 5 else { exit(2) }
    var left = args.count >= 6 ? Int(args[5]) ?? 1 : 1
    var result: AXError?
    walk(app, 0) { e, _ in
        guard args[3] == "*" || text(e, "AXRole") == args[3], label(e).contains(args[4]) else { return false }
        left -= 1
        if left > 0 { return false }
        result = AXUIElementPerformAction(e, "AXPress" as CFString)
        print("press \(text(e, "AXRole")) \(label(e)): \(result!.rawValue)")
        return true
    }
    exit(result == .success ? 0 : 1)
default:
    exit(2)
}
