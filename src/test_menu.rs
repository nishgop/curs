use std::io::{self, Write};

fn main() {
    println!("🔥 Burn Framework Test Menu\n");
    println!("⚠️  NOTE: Burn integration is prepared but not yet active.");
    println!("   There are known compatibility issues with Burn 0.14 + bincode.");
    println!("   Solutions:");
    println!("   1. Use Burn 0.13: cargo add burn@0.13 burn-ndarray@0.13");
    println!("   2. Use WGPU backend: cargo add burn@0.13 --features wgpu");
    println!("   3. Use Candle instead: cargo add candle-core candle-nn\n");
    
    println!("Select a test:");
    println!("1. Show Burn installation commands");
    println!("2. Test current ndarray implementation");
    println!("3. Benchmark current implementation");
    println!("4. Run A* pathfinding demo");
    println!("5. Quick training (10 episodes)");
    println!("0. Exit\n");
    
    print!("Enter choice (0-5): ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    match input.trim() {
        "1" => show_installation(),
        "2" => test_current(),
        "3" => benchmark(),
        "4" => run_astar(),
        "5" => quick_train(),
        "0" => println!("👋 Goodbye!"),
        _ => println!("❌ Invalid choice"),
    }
}

fn show_installation() {
    println!("\n📦 Burn Installation Options:\n");
    println!("Option 1 - Stable (Burn 0.13):");
    println!("  cargo add burn@0.13 --features train,ndarray");
    println!("  cargo add burn-ndarray@0.13\n");
    
    println!("Option 2 - WGPU Backend (Recommended):");
    println!("  cargo add burn@0.13 --features train,wgpu");
    println!("  cargo add burn-wgpu@0.13\n");
    
    println!("Option 3 - Alternative: Candle");
    println!("  cargo add candle-core@0.6");
    println!("  cargo add candle-nn@0.6\n");
}

fn test_current() {
    println!("\n🧪 Testing Current Implementation...");
    println!("Run: cargo run --release -- qtrain 50");
}

fn benchmark() {
    println!("\n⚡ Performance Benchmark");
    println!("Run: time cargo run --release -- qtrain 100");
}

fn run_astar() {
    println!("\n🎮 Running A* Pathfinding Demo...");
    println!("Run: cargo run --release -- astar");
}

fn quick_train() {
    println!("\n🏃 Quick Training Test");
    println!("Run: cargo run --release -- aguidedtrain 10");
}
